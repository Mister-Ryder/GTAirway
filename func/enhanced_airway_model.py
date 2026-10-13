import torch
import torch.nn as nn
import torch.nn.functional as F
from func.model_arch_model10_plus_tou import SegAirwayModel, AirwayClassifier

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv3d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class DomainAdaptiveFusionBlock(nn.Module):
    def __init__(self, seg_channels, cls_channels):
        super(DomainAdaptiveFusionBlock, self).__init__()
        
        # Domain adaptation layers
        self.transform_cls = nn.Sequential(
            nn.Conv3d(cls_channels, seg_channels, kernel_size=1, bias=False),
            nn.InstanceNorm3d(seg_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(seg_channels),
            nn.LeakyReLU(inplace=True)
        )
        
        # Attention mechanisms
        self.channel_attention = ChannelAttention(seg_channels)
        self.spatial_attention = SpatialAttention()
        
        # Feature fusion with gating mechanism
        self.gate = nn.Sequential(
            nn.Conv3d(seg_channels * 2, seg_channels, kernel_size=1),
            nn.InstanceNorm3d(seg_channels),
            nn.Sigmoid()
        )
        
        # Final refinement
        self.refined_conv = nn.Sequential(
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(seg_channels),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, seg_feat, cls_feat):
        # Transform classification features to segmentation space
        adapted_cls_feat = self.transform_cls(cls_feat)
        
        # Apply attention mechanisms to segmentation features
        channel_weights = self.channel_attention(seg_feat)
        attended_seg_feat = seg_feat * channel_weights
        spatial_weights = self.spatial_attention(attended_seg_feat)
        attended_seg_feat = attended_seg_feat * spatial_weights
        
        # Dynamic feature fusion with gating
        concat_feats = torch.cat([attended_seg_feat, adapted_cls_feat], dim=1)
        gate_weights = self.gate(concat_feats)
        
        # Weighted fusion
        fused_features = attended_seg_feat * gate_weights + adapted_cls_feat * (1 - gate_weights)
        
        # Final refinement
        refined_features = self.refined_conv(fused_features)
        
        # Residual connection
        output = refined_features + seg_feat
        
        return output


class MultiScaleFeatureFusion(nn.Module):
    def __init__(self, in_channels, out_channels=None, reduction=2):
        super(MultiScaleFeatureFusion, self).__init__()
        # If out_channels is not provided, use in_channels
        self.out_channels = out_channels if out_channels is not None else in_channels

        self.down_channel = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1)
        # Change up_channel to match the target feature dimensions
        self.up_channel = nn.Conv3d(in_channels // reduction, self.out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, higher_feat, lower_feat):
        # Get the number of channels in the lower feature
        _, c_lower, _, _, _ = lower_feat.size()

        # Down-scale higher resolution feature
        down_feat = self.down_channel(higher_feat)

        # Resize to match lower resolution feature
        _, _, h_lower, w_lower, d_lower = lower_feat.size()
        down_feat = F.adaptive_avg_pool3d(down_feat, (h_lower, w_lower, d_lower))

        # Generate attention weights with correct channel dimension
        attn = self.up_channel(down_feat)
        attn = self.sigmoid(attn)

        # Apply attention to lower feature
        refined_lower_feat = lower_feat * attn

        return refined_lower_feat

class EnhancedAirwaySegmentation(nn.Module):
    def __init__(self, base_segmentation_model, classifier_model):
        super(EnhancedAirwaySegmentation, self).__init__()
        # Load pre-trained models
        self.base_model = base_segmentation_model
        self.classifier = classifier_model

        # Freeze classifier weights (optional, can be adjusted based on training strategy)
        for param in self.classifier.parameters():
            param.requires_grad = False

        # Domain adaptive fusion blocks for each level
        self.fusion_blocks = nn.ModuleList([
            DomainAdaptiveFusionBlock(256, 256),  # Bottleneck
            DomainAdaptiveFusionBlock(128, 128),  # Decoder level 1
            DomainAdaptiveFusionBlock(64, 64),    # Decoder level 2
            DomainAdaptiveFusionBlock(32, 32)     # Decoder level 3
        ])
        
        # Multi-scale feature fusion for encoder-decoder connections
        self.multi_scale_fusion = nn.ModuleList([
            MultiScaleFeatureFusion(in_channels=128, out_channels=64),
            MultiScaleFeatureFusion(in_channels=64, out_channels=32),
            MultiScaleFeatureFusion(in_channels=32, out_channels=32)
        ])

    def forward(self, x):
        # Get classification features and predict class
        cls_logits, cls_features = self.classifier(x)
        cls_features = list(reversed(cls_features))  # Reverse to match encoder-decoder order
        
        # Run through base model's encoders
        encoders_features = []
        for idx, encoder in enumerate(self.base_model.encoders):
            x = encoder(x)
            # Reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)
            
        # Apply domain adaptive fusion to bottleneck features
        bottleneck = self.fusion_blocks[0](encoders_features[0], cls_features[0])
        
        # Enhanced features for decoder path
        encoders_features_enhanced = [bottleneck]
        for i in range(1, len(encoders_features)):
            # Apply domain adaptive fusion to encoder features
            enhanced_encoder_feat = self.fusion_blocks[i](encoders_features[i], cls_features[i])
            encoders_features_enhanced.append(enhanced_encoder_feat)
        
        # Remove the bottleneck for decoder processing
        x_bottleneck = encoders_features_enhanced[0]
        encoders_features_enhanced = encoders_features_enhanced[1:]
            
        # Process through decoders
        decoders_features = []
        x_center_decoders_features = []
        
        # Main decoder path (preserve structure from original model)
        x1 = x_bottleneck
        x2 = x_bottleneck  # For center branch
        
        for i, (decoder, encoder_features, decoder_c) in enumerate(zip(
                self.base_model.decoders, 
                encoders_features_enhanced, 
                self.base_model.decoders_center)):
            
            # Multi-scale feature fusion between decoder levels (if not at the first level)
            if i > 0:
                encoder_features = self.multi_scale_fusion[i-1](decoders_features[-1], encoder_features)
            
            # Decoder processing
            x1 = decoder(encoder_features, x1)
            decoders_features.append(x1)
            
            # Center decoder branch (preserved from original model)
            x2 = decoder_c(encoder_features, x2)
            x_center_decoders_features.append(x2)
        
        # Process final outputs using original model components
        encoders_fusion = self.base_model.dropout1(self.base_model.fusion_module_encoder(encoders_features))
        decoders_fusion = self.base_model.dropout2(self.base_model.fusion_module_decoder(decoders_features))
        decoders_fusion_center = self.base_model.dropout3(
            self.base_model.fusion_module_decoder_center(x_center_decoders_features))
        
        # Generate final outputs
        x_label = self.base_model.final_conv(encoders_fusion)
        x_label = self.base_model.final_activation(x_label)
        
        x_label2 = self.base_model.final_conv2(decoders_fusion)
        x_label2 = self.base_model.final_activation2(x_label2)
        
        # Preserve the Gaussian output (high center connectivity)
        x_gau = self.base_model.final_conv3(decoders_fusion_center)
        x_gau = self.base_model.final_activation3(x_gau)
        
        return x_label, x_label2, x_gau, cls_logits
