A800
export PATH="/home/ubuntu/anaconda3/bin:$PATH"
source activate cu122py39
cd ICME/
python3.9 train_skeleton_model10-ronghe_classfier.py
#taskset -c 0,1 python train_skeleton_model10-ronghe_classfier.py


du -h --max-depth=1


# 实验结论记录
training-script  
版本1:  
发现：  
从目前的结果发现dice显著提升，仅需3个epoch，但其他指标下降，dice本身与bdtl等指标趋势相反。dice的推理很慢，一晚上都还没全部推理玩（70个样本）   
思考：
1. seg2分割损失基本平缓，损失设计上不够合理，要能直接干预bd和tl能力的损失。
2. 周期性更新dice还没有测试，是否有必要。
