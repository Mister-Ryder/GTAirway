# train_anatomical_classifier_unified.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse

from func.load_dataset import airway_dataset, PKLtoNPYAdapter, ResetableRandomSampler, seed_everything
from func.AnatomicalClassifier import AirwayAnatomicalClassifier

# Focal Loss implementation for handling class imbalance more effectively
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        CE_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-CE_loss)
        F_loss = (1 - pt) ** self.gamma * CE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# Custom weighted sampler to balance classes
class ClassBalancedSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, indices=None, num_samples=None, class_key='generation_type', 
                 class_weights=None, replacement=True, generator=None):
        self.dataset = dataset
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples
        self.replacement = replacement
        self.generator = generator
        self.class_key = class_key
        
        # Count instances of each class
        self.class_counts = {}
        for idx in self.indices:
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            if class_value not in self.class_counts:
                self.class_counts[class_value] = 0
            self.class_counts[class_value] += 1
        
        # Calculate weights per class - less frequent classes get higher weight
        if class_weights is None:
            total_samples = float(sum(self.class_counts.values()))
            class_weights = {cls: total_samples / count for cls, count in self.class_counts.items()}
            
            # Increase weight for non-zero classes to counter dominance of "no airway" class
            for cls in class_weights:
                if cls > 0:  # Non-zero classes
                    class_weights[cls] *= 3.0  # Boost non-zero classes
        
        self.class_weights = class_weights
        
        # Assign weight to each sample based on its class
        self.weights = []
        for idx in self.indices:
            class_value = self.dataset.data_dict[self.dataset.name_list[idx]].get(self.class_key, 0)
            self.weights.append(self.class_weights[class_value])
        
        self.weights = torch.DoubleTensor(self.weights)
    
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=self.replacement, generator=self.generator))
    
    def __len__(self):
        return self.num_samples

# Warm-up learning rate scheduler
class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)
        
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
        
    def step(self, epoch=None):
        if self.finished and self.after_scheduler:
            return self.after_scheduler.step(epoch)
        else:
            return super().step(epoch)

def parse_args():
    parser = argparse.ArgumentParser(description='Unified Airway Anatomical Classifier Training')
    
    # Training strategy selection
    parser.add_argument('--strategy', type=str, default='hybrid',
                       choices=['original', 'improved', 'hybrid'],
                       help='Training strategy (original, improved, or hybrid)')
    
    # Common settings
    parser.add_argument('--seed', type=int, default=1419, help='Random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--max_epoch', type=int, default=100, help='Maximum epochs')
    parser.add_argument('--num_samples', type=int, default=12000, help='Samples per epoch')
    parser.add_argument('--save_freq', type=int, default=5, help='Model save frequency (epochs)')
    
    # Dataset settings
    parser.add_argument('--pkl_path', type=str, 
                       default='./dataset_info/airway_metadata_train_anatomical.pkl',
                       help='Path to the metadata pickle file')
    parser.add_argument('--npy_base_dir', type=str,
                       default='/home/ubuntu/shared-data/Precrop_dataset/Precrop_dataset_for_BAS',
                       help='Base directory for NPY files')
    parser.add_argument('--cache_dir', type=str,
                       default='/home/ubuntu/shared-data/BAS/train/.airway_anatomical_cache',
                       help='Cache directory')
    
    # Model checkpoint
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--checkpoint', type=str, 
                       default='/home/ubuntu/ICME/runs_anatomical_classifier/20250401-134306/airway_anatomical_classifier_epoch30.pkl',
                       help='Path to checkpoint to resume from')
    
    # Advanced training settings
    parser.add_argument('--use_focal_loss', action='store_true', help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--use_balanced_sampler', action='store_true', help='Use class-balanced sampler')
    parser.add_argument('--use_warmup', action='store_true', help='Use learning rate warmup')
    parser.add_argument('--filter_empty', action='store_true', help='Filter batches with no airways')
    parser.add_argument('--filter_empty_epoch', type=int, default=33,
                       help='Epoch to start filtering empty airway batches')
    parser.add_argument('--hyper_params_transition_epoch', type=int, default=33,
                       help='Epoch to transition from original to improved hyperparameters (hybrid strategy)')
    
    args = parser.parse_args()
    
    # Set related flags based on strategy
    if args.strategy == 'improved':
        args.use_focal_loss = True
        args.use_balanced_sampler = True
        args.use_warmup = True
        args.filter_empty = True
    elif args.strategy == 'hybrid':
        # Hybrid strategy uses flags as configured
        pass
    
    return args

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    seed_everything(args.seed)

    # Training parameters
    current_time = time.strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join('runs_anatomical_classifier', f'{current_time}_{args.strategy}')
    writer = SummaryWriter(log_dir=log_dir)

    model_name = f"airway_anatomical_classifier_{args.strategy}"
    windowMin_CT_img_HU = -1000
    windowMax_CT_img_HU = 600
    train_file_format = '.npy'
    num_workers = 4
    prefetch_factor = 16
    crop_size = (128, 128, 128)
    
    print(f"Training with strategy: {args.strategy}")
    print(f"Settings: focal_loss={args.use_focal_loss}, balanced_sampler={args.use_balanced_sampler}, "
          f"warmup={args.use_warmup}, filter_empty={args.filter_empty}")

    # Initialize model
    model = AirwayAnatomicalClassifier(
        in_channels=1,
        generation_classes=7,
        structure_classes=7,
        location_classes=7
    )

    # Load checkpoint if resuming training
    start_epoch = 0
    if args.resume and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint)
        
        # Handle both DataParallel and non-DataParallel saved models
        if 'module' in list(checkpoint['model_state_dict'].keys())[0]:
            # Model was saved as DataParallel
            model = nn.DataParallel(model)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Model was saved without DataParallel
            model.load_state_dict(checkpoint['model_state_dict'])
            
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer selection based on strategy
    if args.strategy == 'original' or (args.strategy == 'hybrid' and start_epoch < args.hyper_params_transition_epoch):
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    # Loss function selection
    if args.use_focal_loss:
        # Focal Loss with custom weights
        generation_weights = torch.tensor([0.1, 1.0, 2.5, 2.0, 1.0, 0.8, 2.5]).to(device)
        structure_weights = torch.tensor([0.1, 2.5, 1.5, 2.0, 1.5, 0.8, 1.8]).to(device)
        location_weights = torch.tensor([0.1, 1.8, 1.0, 1.5, 1.0, 1.0, 1.0]).to(device)

        generation_criterion = FocalLoss(alpha=generation_weights, gamma=2.0)
        structure_criterion = FocalLoss(alpha=structure_weights, gamma=2.0)
        location_criterion = FocalLoss(alpha=location_weights, gamma=2.0)
    else:
        # Original CrossEntropyLoss with weights
        generation_weights = torch.tensor([0.2, 1.0, 2.0, 1.5, 1.0, 0.8, 2.0]).to(device)
        structure_weights = torch.tensor([0.2, 2.0, 1.2, 1.5, 1.2, 0.8, 1.5]).to(device)
        location_weights = torch.tensor([0.2, 1.5, 1.0, 1.2, 1.0, 1.0, 1.0]).to(device)

        generation_criterion = nn.CrossEntropyLoss(weight=generation_weights)
        structure_criterion = nn.CrossEntropyLoss(weight=structure_weights)
        location_criterion = nn.CrossEntropyLoss(weight=location_weights)

    # Learning rate scheduler
    if args.use_warmup:
        main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        
        scheduler = GradualWarmupScheduler(
            optimizer, multiplier=1.0, total_epoch=5, after_scheduler=main_scheduler)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Data loading
    adapter = PKLtoNPYAdapter(args.pkl_path, args.npy_base_dir)
    data_dict = adapter.get_adapted_dict()
    print("Adapter conversion complete")

    # Create cache directory
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # Create dataset
    train_dataset = airway_dataset(data_dict, cache_dir=args.cache_dir,
                                  add_anatomical_features=True, model_type="classifier")
    train_dataset.set_para(
        file_format=train_file_format,
        crop_size=crop_size,
        windowMin=windowMin_CT_img_HU,
        windowMax=windowMax_CT_img_HU,
        need_tensor_output=True,
        need_transform=True
    )

    # Create sampler based on strategy
    if args.use_balanced_sampler:
        sampler = ClassBalancedSampler(
            train_dataset,
            num_samples=min(args.num_samples, len(data_dict)),
            class_key='generation_type',
            replacement=True
        )
    else:
        sampler = ResetableRandomSampler(
            train_dataset,
            num_samples=min(args.num_samples, len(data_dict)),
            replacement=False
        )

    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        sampler=sampler
    )

    # Training loop
    start_time = time.time()
    best_loss = float('inf')

    # Save configuration to log directory
    with open(os.path.join(log_dir, 'config.txt'), 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    for epoch in range(start_epoch, args.max_epoch):
        # Check for strategy transition in hybrid mode
        if args.strategy == 'hybrid' and epoch == args.hyper_params_transition_epoch:
            print(f"Transitioning to improved hyperparameters at epoch {epoch}")
            
            # Switch optimizer if needed
            if not isinstance(optimizer, torch.optim.AdamW):
                # Save current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                # Create new optimizer
                optimizer = torch.optim.AdamW(model.parameters(), lr=current_lr, weight_decay=1e-4)
                
                # Update scheduler
                main_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode='min', factor=0.5, patience=10, verbose=True)
                
                scheduler = GradualWarmupScheduler(
                    optimizer, multiplier=1.0, total_epoch=5, after_scheduler=main_scheduler)
            
            # Switch loss functions if needed
            if not args.use_focal_loss:
                args.use_focal_loss = True
                # Focal Loss with custom weights
                generation_weights = torch.tensor([0.1, 1.0, 2.5, 2.0, 1.0, 0.8, 2.5]).to(device)
                structure_weights = torch.tensor([0.1, 2.5, 1.5, 2.0, 1.5, 0.8, 1.8]).to(device)
                location_weights = torch.tensor([0.1, 1.8, 1.0, 1.5, 1.0, 1.0, 1.0]).to(device)

                generation_criterion = FocalLoss(alpha=generation_weights, gamma=2.0)
                structure_criterion = FocalLoss(alpha=structure_weights, gamma=2.0)
                location_criterion = FocalLoss(alpha=location_weights, gamma=2.0)
            
            # Enable empty batch filtering
            args.filter_empty = True
            
            # Log the transition
            writer.add_text('Training/strategy_transition', 
                           f"Transitioned to improved hyperparameters at epoch {epoch}", epoch)

        # Set the epoch for sampler randomization
        if hasattr(sampler, 'set_epoch'):
            sampler.set_epoch(epoch)

        # Train mode
        model.train()
        running_loss = 0.0
        gen_correct = 0
        struct_correct = 0
        loc_correct = 0
        total = 0

        # Track performance for each class
        gen_class_correct = {i: 0 for i in range(7)}
        gen_class_total = {i: 0 for i in range(7)}
        struct_class_correct = {i: 0 for i in range(7)}
        struct_class_total = {i: 0 for i in range(7)}
        loc_class_correct = {i: 0 for i in range(7)}
        loc_class_total = {i: 0 for i in range(7)}

        for batch_idx, batch in enumerate(train_loader):
            # Apply filtering for empty airway batches if enabled
            if args.filter_empty and epoch > args.filter_empty_epoch:
                has_airway = (batch['generation_type'] > 0).sum()
                if has_airway == 0:  # Skip if batch has no airways
                    continue

            # Get inputs and targets
            inputs = batch['image'].float().to(device, non_blocking=True)
            gen_targets = batch['generation_type'].long().to(device, non_blocking=True)
            struct_targets = batch['structure_type'].long().to(device, non_blocking=True)
            loc_targets = batch['location_type'].long().to(device, non_blocking=True)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            gen_logits = outputs['generation']
            struct_logits = outputs['structure']
            loc_logits = outputs['location']

            # Calculate losses
            gen_loss = generation_criterion(gen_logits, gen_targets)
            struct_loss = structure_criterion(struct_logits, struct_targets)
            loc_loss = location_criterion(loc_logits, loc_targets)

            # Combine losses with weighting based on strategy
            if args.strategy == 'original' or (args.strategy == 'hybrid' and epoch < args.hyper_params_transition_epoch):
                # Original weighting
                lambda_gen = 1.0
                lambda_struct = min(1.0, 0.2 + epoch * 0.01)
                lambda_loc = min(1.0, 0.2 + epoch * 0.01)
            else:
                # Improved weighting with focus on generation
                lambda_gen = 1.0
                lambda_struct = min(1.0, 0.5 + epoch * 0.01)
                lambda_loc = min(1.0, 0.5 + epoch * 0.01)
            
            loss = lambda_gen * gen_loss + lambda_struct * struct_loss + lambda_loc * loc_loss

            # Backward pass
            loss.backward()
            
            # Gradient clipping if using improved or hybrid strategy
            if args.strategy != 'original' or (args.strategy == 'hybrid' and epoch >= args.hyper_params_transition_epoch):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Statistics
            running_loss += loss.item()

            # Calculate accuracy
            _, gen_pred = gen_logits.max(1)
            _, struct_pred = struct_logits.max(1)
            _, loc_pred = loc_logits.max(1)

            total += inputs.size(0)
            gen_correct += gen_pred.eq(gen_targets).sum().item()
            struct_correct += struct_pred.eq(struct_targets).sum().item()
            loc_correct += loc_pred.eq(loc_targets).sum().item()

            # Track per-class performance
            for i in range(7):
                gen_mask = (gen_targets == i)
                gen_class_total[i] += gen_mask.sum().item()
                if gen_mask.sum() > 0:
                    gen_class_correct[i] += (gen_pred[gen_mask] == i).sum().item()

                struct_mask = (struct_targets == i)
                struct_class_total[i] += struct_mask.sum().item()
                if struct_mask.sum() > 0:
                    struct_class_correct[i] += (struct_pred[struct_mask] == i).sum().item()

                loc_mask = (loc_targets == i)
                loc_class_total[i] += loc_mask.sum().item()
                if loc_mask.sum() > 0:
                    loc_class_correct[i] += (loc_pred[loc_mask] == i).sum().item()

            # Add to TensorBoard
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train', loss.item(), step)
            writer.add_scalar('Loss/generation', gen_loss.item(), step)
            writer.add_scalar('Loss/structure', struct_loss.item(), step)
            writer.add_scalar('Loss/location', loc_loss.item(), step)
            print(
                f"Epoch: [{epoch + 1}/{args.max_epoch}] "
                f"Batch: [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} "
                )

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                elapsed = time.time() - start_time
                print(
                    f"Epoch: [{epoch + 1}/{args.max_epoch}] "
                    f"Batch: [{batch_idx + 1}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Gen Acc: {100. * gen_correct / total:.2f}% "
                    f"Struct Acc: {100. * struct_correct / total:.2f}% "
                    f"Loc Acc: {100. * loc_correct / total:.2f}% "
                    f"Time: {elapsed:.2f}s"
                )

                writer.add_scalar('Accuracy/generation', 100. * gen_correct / total, step)
                writer.add_scalar('Accuracy/structure', 100. * struct_correct / total, step)
                writer.add_scalar('Accuracy/location', 100. * loc_correct / total, step)

        # Calculate epoch statistics
        avg_loss = running_loss / len(train_loader)
        gen_acc = 100. * gen_correct / total
        struct_acc = 100. * struct_correct / total
        loc_acc = 100. * loc_correct / total

        # Update learning rate
        if isinstance(scheduler, GradualWarmupScheduler):
            scheduler.step(avg_loss)  # warmup scheduler expects a value
        else:
            scheduler.step(avg_loss)

        # Print epoch results
        print(
            f"Epoch: {epoch + 1}/{args.max_epoch} "
            f"Loss: {avg_loss:.4f} "
            f"Gen Acc: {gen_acc:.2f}% "
            f"Struct Acc: {struct_acc:.2f}% "
            f"Loc Acc: {loc_acc:.2f}% "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Log per-class accuracy
        for i in range(7):
            if gen_class_total[i] > 0:
                gen_class_acc = 100. * gen_class_correct[i] / gen_class_total[i]
                writer.add_scalar(f'Accuracy/generation_class_{i}', gen_class_acc, epoch)
                print(f"Gen Class {i} Acc: {gen_class_acc:.2f}% (Total: {gen_class_total[i]})")
                
            if struct_class_total[i] > 0:
                struct_class_acc = 100. * struct_class_correct[i] / struct_class_total[i]
                writer.add_scalar(f'Accuracy/structure_class_{i}', struct_class_acc, epoch)
                print(f"Struct Class {i} Acc: {struct_class_acc:.2f}% (Total: {struct_class_total[i]})")
                
            if loc_class_total[i] > 0:
                loc_class_acc = 100. * loc_class_correct[i] / loc_class_total[i]
                writer.add_scalar(f'Accuracy/location_class_{i}', loc_class_acc, epoch)
                print(f"Loc Class {i} Acc: {loc_class_acc:.2f}% (Total: {loc_class_total[i]})")

        # Add to TensorBoard
        writer.add_scalar('Epoch/loss', avg_loss, epoch)
        writer.add_scalar('Epoch/gen_accuracy', gen_acc, epoch)
        writer.add_scalar('Epoch/struct_accuracy', struct_acc, epoch)
        writer.add_scalar('Epoch/loc_accuracy', loc_acc, epoch)
        writer.add_scalar('Epoch/learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Save model
        if (epoch + 1) % args.save_freq == 0:
            model_path = os.path.join(log_dir, f"{model_name}_epoch{epoch + 1}.pkl")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) 
                                   else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'args': args  # Save configuration
            }, model_path)
            print(f"Model saved to {model_path}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(log_dir, f"{model_name}_best.pkl")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict() if not isinstance(model, nn.DataParallel) 
                                   else model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'args': args  # Save configuration
            }, best_model_path)
            print(f"Best model saved to {best_model_path}")

    # Close SummaryWriter
    writer.close()
    print("Training complete!")


if __name__ == "__main__":
    main()