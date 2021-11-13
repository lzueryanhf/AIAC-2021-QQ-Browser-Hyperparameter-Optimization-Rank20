# AIAC-2021-QQ-Browser-Hyperparameter-Optimization-Rank20

QQ浏览器2021AI算法大赛  赛道二  自动超参数优化  Rank20

新手菜鸡第一次打比赛，全程感觉自己非常幸运，初赛决赛都是踩线进的，成功摸奖！没有像前排大佬采用高级优化方法，大部分时间都用在提升baseline。

## 赛题官网：https://algo.browser.qq.com/

赛题描述：在信息流推荐业务场景中普遍存在模型或策略效果依赖于“超参数”的问题，而“超参数"的设定往往依赖人工经验调参，不仅效率低下维护成本高，而且难以实现更优效果。因此，本次赛题以超参数优化为主题，从真实业务场景问题出发，并基于脱敏后的数据集来评测各个参赛队伍的超参数优化算法。本赛题为超参数优化问题或黑盒优化问题：给定超参数的取值空间，每一轮可以获取一组超参数对应的Reward，要求超参数优化算法在限定的迭代轮次内找到Reward尽可能大的一组超参数，最终按照找到的最大Reward来计算排名。

## 总结
赛题非常新颖，之前没做过超参数的优化，大部分时间花在对赛题的理解上，采用的方法仍旧是baseline的方法，贝叶斯优化+高斯过程回归，上分突破点主要体现在对高斯过程回归部分参数进行优化。初赛训练阶段0.59+，竞技阶段0.51+。

决赛最开始提交的算法仍旧延续初赛思路，将初赛调整好的模型参数迁移到决赛baseline上面，线上大概能取得0.3的成绩。由于自己模型优化算法并不高效，最后并没有采用任何早停算法，使用官方早停算法也没有任何提升。决赛模型仍旧是贝叶斯优化+高斯过程回归，一个重要上分点主要体现在优化初始化抽样，采用了固定初始化的拉丁超立方抽样方法。最终实现训练阶段0.42+，竞技阶段0.39+，由于采用了固定初始化，最终结果还是比较稳定。期间也有尝试过采用组合ucb、ei、poi三种效用函数进行采样优化，但是线上线下结果并不理想，最终放弃该策略。

## 参考
1、https://leovan.me/cn/2020/06/bayesian-optimization/

2、https://zhuanlan.zhihu.com/p/29779000

3、https://www.cnblogs.com/marsggbo/p/10242962.html

4、https://www.cnblogs.com/wmx24/p/10025600.html

5、https://blog.csdn.net/xys430381_1/article/details/104856902

6、https://my.oschina.net/u/4351258/blog/3824759

7、https://blog.csdn.net/jiangshen2009/category_10908162.html
