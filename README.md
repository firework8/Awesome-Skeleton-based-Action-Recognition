# Awesome Skeleton-based Action Recognition  

[![Awesome](https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg)](https://github.com/sindresorhus/awesome)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

We collect existing papers on skeleton-based action recognition published in prominent conferences and journals. 

This paper list will be continuously updated at the end of each month. 


## Table of Contents

- [Survey](#survey)
- [Papers](#papers)
  - [2023](#2023)
  - [2022](#2022)
  - [2021](#2021)
  - [2020](#2020)
  - [2019](#2019)
  - [2018](#2018)
  - [2017](#2017)
  - [2016](#2016)
  - [2015](#2015)
  - [2014](#2014)
- [Other Resources](#other-resources)

## Survey

- Human Action Recognition from Various Data Modalities: A Review (**TPAMI 2022**) [[paper](https://ieeexplore.ieee.org/abstract/document/9795869)]
- Human action recognition and prediction: A survey (**IJCV 2022**) [[paper](https://link.springer.com/article/10.1007/s11263-022-01594-9)]
- Transformer for Skeleton-based action recognition: A review of recent advances (**Neurocomputing 2023**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231223002217)]
- Action recognition based on RGB and skeleton data sets: A survey (**Neurocomputing 2022**) [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222011596)]
- ANUBIS: Review and Benchmark Skeleton-Based Action Recognition Methods with a New Dataset (**2022 arXiv paper**) [[paper](https://arxiv.org/abs/2205.02071)]
- A Survey on 3D Skeleton-Based Action Recognition Using Learning Method (**2020 arXiv paper**) [[paper](https://arxiv.org/abs/2002.05907)]
- A Comparative Review of Recent Kinect-based Action Recognition Algorithms (**TIP 2019**) [[paper](https://ieeexplore.ieee.org/abstract/document/8753686)]

## Papers

Statistics: :fire: highly cited | :star: code is available and star > 100

### 2023

**ACM MM**
- Prompted Contrast with Masked Motion Modeling: Towards Versatile 3D Action Representation Learning [[paper](https://arxiv.org/pdf/2308.03975.pdf)] [[code](https://jhang2020.github.io/Projects/PCM3/PCM3.html)]
- Zero-shot Skeleton-based Action Recognition via Mutual Information Estimation and Maximization [[paper](https://arxiv.org/pdf/2308.03950.pdf)] [[code](https://github.com/YujieOuO/SMIE)]
- Skeleton-MixFormer: Multivariate Topology Representation for Skeleton-based Action Recognition [[code](https://github.com/ElricXin/Skeleton-MixFormer)]

**ICCV**
- Hierarchically Decomposed Graph Convolutional Networks for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Hierarchically_Decomposed_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_ICCV_2023_paper.pdf)] [[code](https://github.com/Jho-Yonsei/HD-GCN)]
- Leveraging Spatio-Temporal Dependency for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lee_Leveraging_Spatio-Temporal_Dependency_for_Skeleton-Based_Action_Recognition_ICCV_2023_paper.pdf)] [[code](https://github.com/Jho-Yonsei/STC-Net)]
- Generative Action Description Prompts for Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Xiang_Generative_Action_Description_Prompts_for_Skeleton-based_Action_Recognition_ICCV_2023_paper.pdf)] [[code](https://github.com/MartinXM/GAP)]
- Masked Motion Predictors are Strong 3D Action Representation Learners [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Mao_Masked_Motion_Predictors_are_Strong_3D_Action_Representation_Learners_ICCV_2023_paper.pdf)] [[code](https://github.com/maoyunyao/MAMP)]
- SkeletonMAE: Graph-based Masked Autoencoder for Skeleton Sequence Pre-training [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yan_SkeletonMAE_Graph-based_Masked_Autoencoder_for_Skeleton_Sequence_Pre-training_ICCV_2023_paper.pdf)] [[code](https://github.com/HongYan1123/SkeletonMAE)]
- MotionBERT: A Unified Perspective on Learning Human Motion Representations [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_MotionBERT_A_Unified_Perspective_on_Learning_Human_Motion_Representations_ICCV_2023_paper.pdf)] [[code](https://github.com/Walter0807/MotionBERT)]
- Parallel Attention Interaction Network for Few-Shot Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_Parallel_Attention_Interaction_Network_for_Few-Shot_Skeleton-Based_Action_Recognition_ICCV_2023_paper.pdf)] [[code](https://github.com/starrycos/PAINet)]
- Modeling the Relative Visual Tempo for Self-supervised Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Zhu_Modeling_the_Relative_Visual_Tempo_for_Self-supervised_Skeleton-based_Action_Recognition_ICCV_2023_paper.pdf)] [[code](https://github.com/Zhuysheng/RVTCLR)]
- FSAR: Federated Skeleton-based Action Recognition with Adaptive Topology Structure and Knowledge Distillation [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Guo_FSAR_Federated_Skeleton-based_Action_Recognition_with_Adaptive_Topology_Structure_and_ICCV_2023_paper.pdf)] [[code](https://github.com/DivyaGuo/FSAR)]
- Hard No-Box Adversarial Attack on Skeleton-Based Human Action Recognition with Skeleton-Motion-Informed Gradient [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Lu_Hard_No-Box_Adversarial_Attack_on_Skeleton-Based_Human_Action_Recognition_with_ICCV_2023_paper.pdf)] [[code](https://github.com/luyg45/HardNoBoxAttack)]
- LAC - Latent Action Composition for Skeleton-based Action Segmentation [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Yang_LAC_-_Latent_Action_Composition_for_Skeleton-based_Action_Segmentation_ICCV_2023_paper.pdf)] [[code](https://github.com/walker1126/Latent_Action_Composition)]
- SkeleTR: Towards Skeleton-based Action Recognition in the Wild [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Duan_SkeleTR_Towards_Skeleton-based_Action_Recognition_in_the_Wild_ICCV_2023_paper.pdf)]
- Cross-Modal Learning with 3D Deformable Attention for Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Cross-Modal_Learning_with_3D_Deformable_Attention_for_Action_Recognition_ICCV_2023_paper.pdf)]

**ICML**
- Ske2Grid: Skeleton-to-Grid Representation Learning for Action Recognition [[paper](http://proceedings.mlr.press/v202/cai23c/cai23c.pdf)] [[code](https://github.com/OSVAI/Ske2Grid)]

**CVPR**
- Learning Discriminative Representations for Skeleton Based Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhou_Learning_Discriminative_Representations_for_Skeleton_Based_Action_Recognition_CVPR_2023_paper.pdf)] [[code](https://github.com/zhysora/FR-Head)]
- Neural Koopman Pooling: Control-Inspired Temporal Dynamics Encoding for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_Neural_Koopman_Pooling_Control-Inspired_Temporal_Dynamics_Encoding_for_Skeleton-Based_Action_CVPR_2023_paper.pdf)] [[code](https://github.com/Infinitywxh/Neural_Koopman_pooling)]
- Actionlet-Dependent Contrastive Learning for Unsupervised Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Lin_Actionlet-Dependent_Contrastive_Learning_for_Unsupervised_Skeleton-Based_Action_Recognition_CVPR_2023_paper.pdf)] [[code](https://langlandslin.github.io/projects/ActCLR/)]
- HaLP: Hallucinating Latent Positives for Skeleton-based Self-Supervised Learning of Actions [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Shah_HaLP_Hallucinating_Latent_Positives_for_Skeleton-Based_Self-Supervised_Learning_of_Actions_CVPR_2023_paper.pdf)] [[code](https://github.com/anshulbshah/HaLP)]
- STMT: A Spatial-Temporal Mesh Transformer for MoCap-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhu_STMT_A_Spatial-Temporal_Mesh_Transformer_for_MoCap-Based_Action_Recognition_CVPR_2023_paper.pdf)] [[code](https://github.com/zgzxy001/STMT)]
- 3Mformer: Multi-order Multi-mode Transformer for Skeletal Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_3Mformer_Multi-Order_Multi-Mode_Transformer_for_Skeletal_Action_Recognition_CVPR_2023_paper.pdf)]
- Unified Pose Sequence Modeling [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Foo_Unified_Pose_Sequence_Modeling_CVPR_2023_paper.pdf)]
- Unified Keypoint-based Action Recognition Framework via Structured Keypoint Pooling [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Hachiuma_Unified_Keypoint-Based_Action_Recognition_Framework_via_Structured_Keypoint_Pooling_CVPR_2023_paper.pdf)]
- Prompt-Guided Zero-Shot Anomaly Action Recognition using Pretrained Deep Skeleton Features [[paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Sato_Prompt-Guided_Zero-Shot_Anomaly_Action_Recognition_Using_Pretrained_Deep_Skeleton_Features_CVPR_2023_paper.pdf)]

**ICLR**
- Graph Contrastive Learning for Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2301.10900.pdf)] [[code](https://github.com/OliverHxh/SkeletonGCL)]
- Hyperbolic Self-paced Learning for Self-supervised Skeleton-based Action Representations [[paper](https://arxiv.org/pdf/2303.06242.pdf)] [[code](https://github.com/paolomandica/HYSP)]

**AAAI**
- Hierarchical Consistent Contrastive Learning for Skeleton-Based Action Recognition with Growing Augmentations [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25451)] [[code](https://github.com/JHang2020/HiCLR)]
- Self-supervised Action Representation Learning from Partial Spatio-Temporal Skeleton Sequences [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25495)] [[code](https://github.com/YujieOuO/PSTL)]
- Frame-Level Label Refinement for Skeleton-Based Weakly-Supervised Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25439)] [[code](https://github.com/line/Skeleton-Temporal-Action-Localization)]
- Hierarchical Contrast for Unsupervised Skeleton-based Action Representation Learning [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25127)] [[code](https://github.com/HuiGuanLab/HiCo)]
- Anonymization for Skeleton Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/26754)] [[code](https://github.com/ml-postech/Skeleton-anonymization)]
- Defending Black-box Skeleton-based Human Activity Classifiers [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25352)] [[code](https://github.com/realcrane/Defending-Black-box-Skeleton-based-Human-Activity-Classifiers)]
- Novel Motion Patterns Matter for Practical Skeleton-based Action Recognition [[paper](https://humanperception.github.io/documents/AAAI2023.pdf)]
- Self-Supervised Learning for Multilevel Skeleton-Based Forgery Detection via Temporal-Causal Consistency of Actions [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/25163)]

**IJCAI**
- Part Aware Contrastive Learning for Self-Supervised Action Recognition [[paper](https://www.ijcai.org/proceedings/2023/0095.pdf)] [[code](https://github.com/GitHubOfHyl97/SkeAttnCLR)]
- Action Recognition with Multi-stream Motion Modeling and Mutual Information Maximization [[paper](https://www.ijcai.org/proceedings/2023/0184.pdf)]

**ICCVW**
- A Lightweight Skeleton-Based 3D-CNN for Real-Time Fall Detection and Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2023W/JRDB/papers/Noor_A_Lightweight_Skeleton-Based_3D-CNN_for_Real-Time_Fall_Detection_and_Action_ICCVW_2023_paper.pdf)]

**WACV**
- Adaptive Local-Component-aware Graph Convolutional Network for One-shot Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content/WACV2023/papers/Zhu_Adaptive_Local-Component-Aware_Graph_Convolutional_Network_for_One-Shot_Skeleton-Based_Action_Recognition_WACV_2023_paper.pdf)]

**ICIP**
- Temporal-Channel Topology Enhanced Network for Skeleton-Based Action Recognition [[paper](https://arxiv.org/ftp/arxiv/papers/2302/2302.12967.pdf)] [[code](https://github.com/aikuniverse/TCTE-Net)]
- Part Aware Graph Convolution Network with Temporal Enhancement for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/10222714)]
- Skeleton Action Recognition Based on Spatio-Temporal Features [[paper](https://ieeexplore.ieee.org/abstract/document/10223086)]

**ICME**
- DD-GCN: Directed Diffusion Graph Convolutional Network for Skeleton-based Human Action Recognition [[paper](https://arxiv.org/pdf/2308.12501.pdf)] [[code](https://github.com/shiyin-lc/DD-GCN)]

**ICMEW**
- SkeletonMAE: Spatial-Temporal Masked Autoencoders for Self-supervised Skeleton Action Recognition [[paper](https://arxiv.org/pdf/2209.02399.pdf)]

**IROS**
- Interactive Spatiotemporal Token Attention Network for Skeleton-based General Interactive Action Recognition [[paper](https://arxiv.org/pdf/2307.07469.pdf)] [[code](https://github.com/Necolizer/ISTA-Net)]

**TPAMI**
- Self-Supervised 3D Action Representation Learning with Skeleton Cloud Colorization [[paper](https://arxiv.org/pdf/2304.08799.pdf)]

**TMM**
- Delving Deep into One-Shot Skeleton-based Action Recognition with Diverse Occlusions [[paper](https://ieeexplore.ieee.org/abstract/document/10011561)] [[code](https://github.com/KPeng9510/Trans4SOAR)]
- Temporal Decoupling Graph Convolutional Network for Skeleton-based Gesture Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/10113233)] [[code](https://github.com/liujf69/TD-GCN-Gesture)]
- Skeleton-based Action Recognition through Contrasting Two-Stream Spatial-Temporal Networks [[paper](https://arxiv.org/pdf/2301.11495.pdf)]
- Learning Representations by Contrastive Spatio-temporal Clustering for Skeleton-based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/10227565)]
- Skeleton-Based Gesture Recognition With Learnable Paths and Signature Features [[paper](https://ieeexplore.ieee.org/abstract/document/10261439)]
- Skeleton-Based Action Recognition with Select-Assemble-Normalize Graph Convolutional Networks [[paper](https://ieeexplore.ieee.org/abstract/document/10265127)]

**TCSVT**
- Motion Complement and Temporal Multifocusing for Skeleton-based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/10015806)] [[code](https://github.com/cong-wu/MCMT-Net)]
- TranSkeleton: Hierarchical Spatial-Temporal Transformer for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/10029908)]

**TNNLS**
- Spatiotemporal Decouple-and-Squeeze Contrastive Learning for Semi-Supervised Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2302.02316.pdf)]
- Learning Heterogeneous Spatial–Temporal Context for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/10081331)]
- Self-Adaptive Graph With Nonlocal Attention Network for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/10250900)]

**PR**
- Continual spatio-temporal graph convolutional networks [[paper](https://www.sciencedirect.com/science/article/pii/S0031320323002285)] [[code](https://github.com/LukasHedegaard/continual-skeletons)]
- Relation-mining self-attention network for skeleton-based human action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0031320323001553)] [[code](https://github.com/GedamuA/RSA-Net)]
- SpatioTemporal Focus for Skeleton-based Action Recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0031320322007105)]

**Neurocomputing**
- Focalized Contrastive View-invariant Learning for Self-supervised Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2304.00858)]
- Spatio-temporal segments attention for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222013716)]
- SPAR: An efficient self-attention network using Switching Partition Strategy for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S092523122301038X)]
- STDM-transformer: Space-time dual multi-scale transformer network for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231223010263)]

**arXiv papers**
- Language Knowledge-Assisted Representation Learning for Skeleton-Based Action Recognition [[paper](https://arxiv.org/abs/2305.12398)] [[code](https://github.com/damnull/lagcn)]
- TSGCNeXt: Dynamic-Static Multi-Graph Convolution for Efficient Skeleton-Based Action Recognition with Long-term Learning Potential [[paper](https://arxiv.org/abs/2304.11631)] [[code](https://github.com/vvhj/TSGCNeXt)]
- Overcoming Topology Agnosticism: Enhancing Skeleton-Based Action Recognition through Redefined Skeletal Topology Awareness [[paper](https://arxiv.org/abs/2305.11468)] [[code](https://github.com/ZhouYuxuanYX/BlockGCN)]
- SiT-MLP: A Simple MLP with Point-wise Topology Feature Learning for Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2308.16018)] [[code](https://github.com/BUPTSJZhang/SiT-MLP)]
- Balanced Representation Learning for Long-tailed Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2308.14024)] [[code](https://github.com/firework8/BRL)]
- Joint Adversarial and Collaborative Learning for Self-Supervised Action Recognition [[paper](https://arxiv.org/abs/2307.07791)] [[code](https://github.com/Levigty/ACL)]
- Unveiling the Hidden Realm: Self-supervised Skeleton-based Action Recognition in Occluded Environments [[paper](https://arxiv.org/abs/2309.12029)] [[code](https://github.com/cyfml/OPSTL)]
- Elevating Skeleton-Based Action Recognition with Efficient Multi-Modality Self-Supervision [[paper](https://arxiv.org/abs/2309.12009)] [[code](https://github.com/desehuileng0o0/IKEM)]
- Pyramid Self-attention Polymerization Learning for Semi-supervised Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2302.02327)] [[code](https://github.com/1xbq1/PSP-Learning)]
- Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2305.17939)] [[code](https://github.com/nntanaka/Fourier-Analysis-for-Skeleton-based-Action-Recognition)]
- InfoGCN++: Learning Representation by Predicting the Future for Online Human Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2310.10547)] [[code](https://github.com/stnoah1/infogcn2)]
- High-Performance Inference Graph Convolutional Networks for Skeleton-Based Action Recognition [[paper](https://arxiv.org/abs/2305.18710)]
- Dynamic Spatial-temporal Hypergraph Convolutional Network for Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2302.08689)]
- Skeleton-based Human Action Recognition via Convolutional Neural Networks (CNN) [[paper](https://arxiv.org/abs/2301.13360)]
- Cross-view Action Recognition via Contrastive View-invariant Representation [[paper](https://arxiv.org/abs/2305.01733)]
- Attack is Good Augmentation: Towards Skeleton-Contrastive Representation Learning [[paper](https://arxiv.org/abs/2304.04023)]
- SCD-Net: Spatiotemporal Clues Disentanglement Network for Self-supervised Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2309.05834)]
- DMMG: Dual Min-Max Games for Self-Supervised Skeleton-Based Action Recognition [[paper](https://arxiv.org/abs/2302.12007)]
- Spatial-temporal Transformer-guided Diffusion based Data Augmentation for Efficient Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2302.13434)]
- Modiff: Action-Conditioned 3D Motion Generation with Denoising Diffusion Probabilistic Models [[paper](https://arxiv.org/abs/2301.03949)]
- Skeleton-based action analysis for ADHD diagnosis [[paper](https://arxiv.org/abs/2304.09751)]
- Multi-Dimensional Refinement Graph Convolutional Network with Robust Decouple Loss for Fine-Grained Skeleton-Based Action Recognition [[paper](https://arxiv.org/abs/2306.15321)]
- Fine-grained Action Analysis: A Multi-modality and Multi-task Dataset of Figure Skating [[paper](https://arxiv.org/abs/2307.02730)]
- Physical-aware Cross-modal Adversarial Network for Wearable Sensor-based Human Action Recognition [[paper](https://arxiv.org/abs/2307.03638)]
- Improving Video Violence Recognition with Human Interaction Learning on 3D Skeleton Point Clouds [[paper](https://arxiv.org/abs/2308.13866)]


### 2022

**CVPR**
- InfoGCN: Representation Learning for Human Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Chi_InfoGCN_Representation_Learning_for_Human_Skeleton-Based_Action_Recognition_CVPR_2022_paper.pdf)] [[code](https://github.com/stnoah1/infogcn)]
- Revisiting Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Duan_Revisiting_Skeleton-Based_Action_Recognition_CVPR_2022_paper.pdf)] [[code](https://github.com/kennymckormick/pyskl)]

**ECCV**
- CMD: Self-supervised 3D Action Representation Learning with Cross-modal Mutual Distillation [[paper](https://arxiv.org/pdf/2208.12448.pdf)] [[code](https://github.com/maoyunyao/CMD)]
- Hierarchically Self-Supervised Transformer for Human Skeleton Representation Learning [[paper](https://arxiv.org/pdf/2207.09644.pdf)] [[code](https://github.com/yuxiaochen1103/Hi-TRS)]
- Collaborating Domain-shared and Target-specific Feature Clustering for Cross-domain 3D Action Recognition [[paper](https://arxiv.org/pdf/2207.09767.pdf)] [[code](https://github.com/canbaoburen/CoDT)]
- Global-local Motion Transformer for Unsupervised Skeleton-based Action Learning [[paper](https://arxiv.org/pdf/2207.06101.pdf)] [[code](https://github.com/Boeun-Kim/GL-Transformer)]
- Contrastive Positive Mining for Unsupervised 3D Action Representation Learning [[paper](https://arxiv.org/pdf/2208.03497.pdf)]
- Learning Spatial-Preserved Skeleton Representations for Few-Shot Action Recognition [[paper](https://openreview.net/pdf?id=qIlLNOJsKxJ)]
- Uncertainty-DTW for Time Series and Sequences [[paper](https://arxiv.org/pdf/2211.00005.pdf)]

**AAAI**
- Contrastive Learning from Extremely Augmented Skeleton Sequences for Self-supervised Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19957)] [[code](https://github.com/Levigty/AimCLR)]
- Topology-aware Convolutional Neural Network for Efficient Skeleton-based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/20191)] [[code](https://github.com/hikvision-research/skelact)]
- Towards To-a-T Spatio-Temporal Focus for Skeleton-Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/19998)]

**ACM MM**
- PYSKL: Towards Good Practices for Skeleton Action Recognition [[paper](https://arxiv.org/pdf/2205.09443.pdf)] [[code](https://github.com/kennymckormick/pyskl)]
- Shifting Perspective to See Difference: A Novel Multi-View Method for Skeleton based Action Recognition [[paper](https://arxiv.org/pdf/2209.02986.pdf)] [[code](https://github.com/ideal-idea/SAP)]
- Skeleton-based Action Recognition via Adaptive Cross-Form Learning [[paper](https://arxiv.org/pdf/2206.15085.pdf)] [[code](https://github.com/stoa-xh91/ACFL)]
- Global-Local Cross-View Fisher Discrimination for View-Invariant Action Recognition [[paper](https://dl.acm.org/doi/abs/10.1145/3503161.3548280)]

**CVPRW**
- Bootstrapped Representation Learning for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2022W/L3D-IVU/papers/Moliner_Bootstrapped_Representation_Learning_for_Skeleton-Based_Action_Recognition_CVPRW_2022_paper.pdf)]

**ECCVW**
- Mitigating Representation Bias in Action Recognition: Algorithms and Benchmarks [[paper](https://arxiv.org/pdf/2209.09393.pdf)] [[code](https://github.com/kennymckormick/ARAS-Dataset)]
- PSUMNet: Unified Modality Part Streams are All You Need for Efficient Pose-based Action Recognition [[paper](https://arxiv.org/pdf/2208.05775.pdf)] [[code](https://github.com/skelemoa/psumnet)]
- Strengthening Skeletal Action Recognizers via Leveraging Temporal Patterns [[paper](https://arxiv.org/pdf/2205.14405.pdf)]

**ACCV**
- Focal and Global Spatial-Temporal Transformer for Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Gao_Focal_and_Global_Spatial-Temporal_Transformer_for_Skeleton-based_Action_Recognition_ACCV_2022_paper.pdf)] 
- Temporal-Viewpoint Transportation Plan for Skeletal Few-shot Action Recognition [[paper](https://openaccess.thecvf.com/content/ACCV2022/papers/Wang_Temporal-Viewpoint_Transportation_Plan_for_Skeletal_Few-shot_Action_Recognition_ACCV_2022_paper.pdf)]

**WACV**
- Skeleton-DML: Deep Metric Learning for Skeleton-Based One-Shot Action Recognition [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Memmesheimer_Skeleton-DML_Deep_Metric_Learning_for_Skeleton-Based_One-Shot_Action_Recognition_WACV_2022_paper.pdf)] [[code](https://github.com/raphaelmemmesheimer/skeleton-dml)]
- Generative Adversarial Graph Convolutional Networks for Human Action Synthesis [[paper](https://openaccess.thecvf.com/content/WACV2022/papers/Degardin_Generative_Adversarial_Graph_Convolutional_Networks_for_Human_Action_Synthesis_WACV_2022_paper.pdf)] [[code](https://github.com/DegardinBruno/Kinetic-GAN)]

**ICPR**
- Skeletal Human Action Recognition using Hybrid Attention based Graph Convolutional Network [[paper](https://arxiv.org/pdf/2207.05493.pdf)]

**TPAMI**
- Constructing Stronger and Faster Baselines for Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2106.15125.pdf)] [[code](https://gitee.com/yfsong0709/EfficientGCNv1)]
- Motif-GCNs With Local and Non-Local Temporal Blocks for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9763364)] [[code](https://github.com/wenyh1616/SAMotif-GCN)]
- Multi-Granularity Anchor-Contrastive Representation Learning for Semi-Supervised Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9954217)] [[code](https://github.com/1xbq1/MAC-Learning)]

**IJCV**
- Action2video: Generating Videos of Human 3D Actions [[paper](https://link.springer.com/article/10.1007/s11263-021-01550-z)]

**TIP**
- Contrast-reconstruction Representation Learning for Self-supervised Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2111.11051.pdf)] [[code](https://github.com/Picasso-Wang/CRRL)]
- Multilevel Spatial–Temporal Excited Graph Network for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9997556)] [[code](https://github.com/Zhuysheng/ML-STGNet)]
- SMAM: Self and Mutual Adaptive Matching for Skeleton-Based Few-Shot Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9975251)]
- X-Invariant Contrastive Augmentation and Representation Learning for Semi-Supervised Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9782720)]

**TMM**
- Skeleton-Based Mutually Assisted Interacted Object Localization and Human Action Recognition [[paper](https://arxiv.org/pdf/2110.14994.pdf)]
- Joint-bone Fusion Graph Convolutional Network for Semi-supervised Skeleton Action Recognition [[paper](https://arxiv.org/ftp/arxiv/papers/2202/2202.04075.pdf)]

**TCSVT**
- Two-person Graph Convolutional Network for Skeleton-based Human Interaction Recognition [[paper](https://arxiv.org/pdf/2208.06174.pdf)] [[code](https://github.com/mgiant/2P-GCN)]
- Zoom Transformer for Skeleton-Based Group Activity Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9845486)] [[code](https://github.com/Kebii/Zoom-Transformer)]
- Motion Guided Attention Learning for Self-Supervised 3D Human Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9841515)]
- Motion-Driven Spatial and Temporal Adaptive High-Resolution Graph Convolutional Networks for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9931755)]
- View-Normalized and Subject-Independent Skeleton Generation for Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9940286)]

**TNNLS**
- Fusing Higher-Order Features in Graph Neural Networks for Skeleton-Based Action Recognition [[paper](https://arxiv.org/pdf/2105.01563.pdf)] [[code](https://github.com/ZhenyueQin/Angular-Skeleton-Encoding)]

**Neurocomputing**
- Forward-reverse adaptive graph convolutional networks for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231221018920)] [[code](https://github.com/Nanasaki-Ai/FR-AGCN)]
- AFE-CNN: 3D Skeleton-based Action Recognition with Action Feature Enhancement [[paper](https://arxiv.org/pdf/2208.03444.pdf)]
- Hierarchical graph attention network with pseudo-metapath for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222007421)]
- Skeleton-based similar action recognition through integrating the salient image feature into a center-connected graph convolutional network [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222009560)]
- PB-GCN: Progressive binary graph convolutional networks for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231222008049)]

**arXiv papers**
- Hypergraph Transformer for Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2211.09590)] [[code](https://github.com/ZhouYuxuanYX/Hypergraph-Transformer-for-Skeleton-based-Action-Recognition)]
- DG-STGCN: Dynamic Spatial-Temporal Modeling for Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2210.05895)] [[code](https://github.com/kennymckormick/pyskl)]
- Spatio-Temporal Tuples Transformer for Skeleton-Based Action Recognition [[paper](https://arxiv.org/abs/2201.02849)] [[code](https://github.com/heleiqiu/STTFormer)]
- Contrastive Learning from Spatio-Temporal Mixed Skeleton Sequences for Self-Supervised Skeleton-Based Action Recognition [[paper](https://arxiv.org/abs/2207.03065)] [[code](https://github.com/czhaneva/SkeleMixCLR)]
- ViA: View-invariant Skeleton Action Representation Learning via Motion Retargeting [[paper](https://arxiv.org/abs/2209.00065)] [[code](https://github.com/YangDi666/UNIK)]
- HAA4D: Few-Shot Human Atomic Action Recognition via 3D Spatio-Temporal Skeletal Alignment [[paper](https://arxiv.org/abs/2202.07308)] [[code](https://github.com/Morris88826/HAA4D)]
- Skeleton-based Action Recognition Via Temporal-Channel Aggregation [[paper](https://arxiv.org/abs/2205.15936)]
- A New Spatial Adjacency Matrix of Skeleton Data Based on Self-loop and Adaptive Weights [[paper](https://arxiv.org/abs/2206.14344)]
- View-Invariant Skeleton-based Action Recognition via Global-Local Contrastive Learning [[paper](https://arxiv.org/abs/2209.11634)]


### 2021

**CVPR**
- 3D Human Action Representation Learning via Cross-View Consistency Pursuit [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_3D_Human_Action_Representation_Learning_via_Cross-View_Consistency_Pursuit_CVPR_2021_paper.pdf)] [[code](https://github.com/LinguoLi/CrosSCLR)]
- BASAR:Black-box Attack on Skeletal Action Recognition [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Diao_BASARBlack-Box_Attack_on_Skeletal_Action_Recognition_CVPR_2021_paper.pdf)] [[code](https://github.com/realcrane/BASAR-Black-box-Attack-on-Skeletal-Action-Recognition)]
- Understanding the Robustness of Skeleton-based Action Recognition under Adversarial Attack [[paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Understanding_the_Robustness_of_Skeleton-Based_Action_Recognition_Under_Adversarial_Attack_CVPR_2021_paper.pdf)] [[code](https://github.com/realcrane/Understanding-the-Robustness-of-Skeleton-based-Action-Recognition-under-Adversarial-Attack)]

**ICCV**
- Channel-wise Topology Refinement Graph Convolution for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Chen_Channel-Wise_Topology_Refinement_Graph_Convolution_for_Skeleton-Based_Action_Recognition_ICCV_2021_paper.pdf)] [[code](https://github.com/Uason-Chen/CTR-GCN)]
- AdaSGN: Adapting Joint Number and Model Size for Efficient Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Shi_AdaSGN_Adapting_Joint_Number_and_Model_Size_for_Efficient_Skeleton-Based_ICCV_2021_paper.pdf)] [[code](https://github.com/lshiwjx/AdaSGN)]
- Skeleton Cloud Colorization for Unsupervised 3D Action Representation Learning [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Skeleton_Cloud_Colorization_for_Unsupervised_3D_Action_Representation_Learning_ICCV_2021_paper.pdf)]
- Self-supervised 3D Skeleton Action Representation Learning with Motion Consistency and Continuity [[paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Su_Self-Supervised_3D_Skeleton_Action_Representation_Learning_With_Motion_Consistency_and_ICCV_2021_paper.pdf)]

**NeurIPS** 
- Unsupervised Motion Representation Learning with Capsule Autoencoders [[paper](https://proceedings.neurips.cc/paper/2021/file/19ca14e7ea6328a42e0eb13d585e4c22-Paper.pdf)] [[code](https://github.com/ZiweiXU/CapsuleMotion)]

**AAAI**
- Multi-Scale Spatial Temporal Graph Convolutional Network for Skeleton-Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16197)] [[code](https://github.com/czhaneva/MST-GCN)]
- Spatio-Temporal Difference Descriptor for Skeleton-Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/16210)]

**ACM MM**
- Learning Multi-Granular Spatio-Temporal Graph Network for Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2108.04536.pdf)] [[code](https://github.com/tailin1009/DualHead-Network)]
- STST: Spatial-Temporal Specialized Transformer for Skeleton-based Action Recognition [[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475473)] [[code](https://github.com/HanzoZY/STST)]
- Skeleton-Contrastive 3D Action Representation Learning [[paper](https://arxiv.org/pdf/2108.03656.pdf)] [[code](https://github.com/fmthoker/skeleton-contrast)]
- Modeling the Uncertainty for Self-supervised 3D Skeleton Action Representation Learning [[paper](https://dl.acm.org/doi/abs/10.1145/3474085.3475248)]

**CVPRW**
- One-shot action recognition in challenging therapy scenarios [[paper](https://openaccess.thecvf.com/content/CVPR2021W/LLID/papers/Sabater_One-Shot_Action_Recognition_in_Challenging_Therapy_Scenarios_CVPRW_2021_paper.pdf)] [[code](https://github.com/AlbertoSabater/Skeleton-based-One-shot-Action-Recognition)]

**BMVC**
- UNIK: A Unified Framework for Real-world Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2107.08580.pdf)] [[code](https://github.com/YangDi666/UNIK)]
- Unsupervised Human Action Recognition with Skeletal Graph Laplacian and Self-Supervised Viewpoints Invariance [[paper](https://arxiv.org/pdf/2204.10312.pdf)] [[code](https://github.com/IIT-PAVIS/UHAR_Skeletal_Laplacian)]
- LSTA-Net: Long short-term Spatio-Temporal Aggregation Network for Skeleton-based Action Recognition [[paper](https://arxiv.org/abs/2111.00823)]

**WACV**
- JOLO-GCN: Mining Joint-Centered Light-Weight Information for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content/WACV2021/papers/Cai_JOLO-GCN_Mining_Joint-Centered_Light-Weight_Information_for_Skeleton-Based_Action_Recognition_WACV_2021_paper.pdf)]

**ICPR**
- Learning Connectivity with Graph Convolutional Networks for Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2112.03328.pdf)]

**ICPRW**
- Spatial Temporal Transformer Network for Skeleton-Based Action Recognition [[paper](https://arxiv.org/pdf/2012.06399.pdf)] [[code](https://github.com/Chiaraplizz/ST-TR)]

**ICIP**
- Syntactically Guided Generative Embeddings for Zero-Shot Skeleton Action Recognition [[paper](https://arxiv.org/pdf/2101.11530.pdf)] [[code](https://github.com/skelemoa/synse-zsl)]

**ICME**
- Graph Convolutional Hourglass Networks for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9428355)]

**ICRA**
- Pose Refinement Graph Convolutional Network for Skeleton-basedAction Recognition [[paper](https://arxiv.org/pdf/2010.07367.pdf)] [[code](https://github.com/sj-li/PR-GCN)]

**TPAMI**
- Symbiotic Graph Neural Networks for 3D Skeleton-Based Human Action Recognition and Motion Prediction [[paper](https://arxiv.org/pdf/1910.02212.pdf)]
- Tensor Representations for Action Recognition [[paper](https://arxiv.org/pdf/2012.14371.pdf)]

**IJCV**
- Quo Vadis, Skeleton Action Recognition? [[paper](https://arxiv.org/pdf/2007.02072.pdf)] [[code](https://skeleton.iiit.ac.in)]

**TIP**
- Extremely Lightweight Skeleton-Based Action Recognition with ShiftGCN++ [[paper](https://ieeexplore.ieee.org/abstract/document/9515708)] [[code](https://github.com/kchengiva/Shift-GCN-plus)]
- Structural Knowledge Distillation for Efficient Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9351789)] [[code](https://github.com/xiaochehe/SKD)]
- Feedback Graph Convolutional Network for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9626596)]
- Hypergraph Neural Network for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9329123)]

**TIFS**
- REGINA - Reasoning Graph Convolutional Networks in Human Action Recognition [[paper](https://arxiv.org/pdf/2105.06711.pdf)] [[code](https://github.com/DegardinBruno)]

**TMM**
- Prototypical Contrast and Reverse Prediction: Unsupervised Skeleton Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9623511)] [[code](https://github.com/LZU-SIAT/PCRP)]
- Interaction Relational Network for Mutual Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9319533)] [[code](https://github.com/mauriciolp/inter-rel-net)]
- LAGA-Net: Local-and-Global Attention Network for Skeleton Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9447926)]
- A Multi-Stream Graph Convolutional Networks-Hidden Conditional Random Field Model for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9000721)]
- Multi-Localized Sensitive Autoencoder-Attention-LSTM For Skeleton-based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9392333)]
- Dear-Net: Learning Diversities for Skeleton-Based Early Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9667321)]
- Efficient Spatio-Temporal Contrastive Learning for Skeleton-Based 3-D Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9612062)]
- GA-Net: A Guidance Aware Network for Skeleton-Based Early Activity Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9661424)]

**TCSVT**
- Fuzzy Integral-Based CNN Classifier Fusion for 3D Skeleton Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9177170)] [[code](https://github.com/theavicaster/fuzzy-integral-cnn-fusion-3d-har)]
- A Central Difference Graph Convolutional Operator for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9597501)] [[code](https://github.com/iesymiao/CD-GCN)]
- Multi-Stream Interaction Networks for Human Action Recognition [[paper](https://ieeexplore.ieee.org/document/9492107)]
- A Cross View Learning Approach for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9496611)]
- Symmetrical Enhanced Fusion Network for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9319717)]
- Graph2Net: Perceptually-enriched graph learning for skeleton-based action recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9446181)]

**TNNLS**
- Memory Attention Networks for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9378801)]  [[code](https://github.com/memory-attention-networks/MANs)]

**PR**
- Arbitrary-view human action recognition via novel-view action generation [[paper](https://www.sciencedirect.com/science/article/pii/S0031320321002302)] [[code](https://github.com/GedamuA/TB-GAN)]
- Tripool: Graph triplet pooling for 3D skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0031320321001084)]
- Action recognition using kinematics posture feature on 3D skeleton joint locations [[paper](https://www.sciencedirect.com/science/article/pii/S0167865521000751)]
- Scene image and human skeleton-based dual-stream human action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0167865521001902)]

**Neurocomputing**
- Rethinking the ST-GCNs for 3D skeleton-based human action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231221007153)]
- Attention adjacency matrix based graph convolutional networks for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231221002101)]
- Skeleton-based action recognition using sparse spatio-temporal GCN with edge effective resistance [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220317094)]
- Integrating vertex and edge features with Graph Convolutional Networks for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231221013928)]
- Adaptive multi-view graph convolutional networks for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220317690)]
- Knowledge embedded GCN for skeleton-based two-person interaction recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220317732)]
- Normal graph: Spatial temporal graph convolutional networks based prediction network for skeleton based video anomaly detection [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220317720)]

**arXiv papers**
- IIP-Transformer: Intra-Inter-Part Transformer for Skeleton-Based Action Recognition [[paper](https://arxiv.org/abs/2110.13385)] [[code](https://github.com/qtwang0035/IIP-Transformer)]
- STAR: Sparse Transformer-based Action Recognition [[paper](https://arxiv.org/abs/2107.07089)] [[code](https://github.com/imj2185/STAR)]
- Self-attention based anchor proposal for skeleton-based action recognition [[paper](https://arxiv.org/abs/2112.09413)] [[code](https://github.com/ideal-idea/SAP)]
- Multi-Scale Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition [[paper](https://arxiv.org/abs/2111.03993)]
- 3D Skeleton-based Few-shot Action Recognition with JEANIE is not so Na¨ıve [[paper](https://arxiv.org/abs/2112.12668)]


### 2020

**CVPR**
- Disentangling and Unifying Graph Convolutions for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Liu_Disentangling_and_Unifying_Graph_Convolutions_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.pdf)] [[code](https://github.com/kenziyuliu/ms-g3d)]
- Skeleton-Based Action Recognition with Shift Graph Convolutional Network [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Cheng_Skeleton-Based_Action_Recognition_With_Shift_Graph_Convolutional_Network_CVPR_2020_paper.pdf)] [[code](https://github.com/kchengiva/Shift-GCN)]
- Semantics-Guided Neural Networks for Efficient Skeleton-Based Human Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Semantics-Guided_Neural_Networks_for_Efficient_Skeleton-Based_Human_Action_Recognition_CVPR_2020_paper.pdf)] [[code](https://github.com/microsoft/SGN)]
- PREDICT & CLUSTER: Unsupervised Skeleton Based Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Su_PREDICT__CLUSTER_Unsupervised_Skeleton_Based_Action_Recognition_CVPR_2020_paper.pdf)] [[code](https://github.com/shlizee/Predict-Cluster)]
- Dynamic Multiscale Graph Neural Networks for 3D Skeleton Based Human Motion Prediction [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Li_Dynamic_Multiscale_Graph_Neural_Networks_for_3D_Skeleton_Based_Human_CVPR_2020_paper.pdf)] [[code](https://github.com/limaosen0/DMGNN)]
- Context Aware Graph Convolution for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Context_Aware_Graph_Convolution_for_Skeleton-Based_Action_Recognition_CVPR_2020_paper.pdf)]

**ECCV**
- Decoupling GCN with DropGraph Module for Skeleton-Based Action Recognition [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58586-0_32)] [[code](https://github.com/kchengiva/DecoupleGCN-DropGraph)]
- Unsupervised 3D Human Pose Representation with Viewpoint and Pose Disentanglement [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58529-7_7)] [[code](https://github.com/NIEQiang001/unsupervised-human-pose)]
- Adversarial Self-supervised Learning for Semi-supervised 3D Action Recognition [[paper](https://link.springer.com/chapter/10.1007/978-3-030-58571-6_3)]

**AAAI**
- Learning Graph Convolutional Network for Skeleton-based Human Action Recognition by Neural Searching [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/5652)] [[code](https://github.com/xiaoiker/GCN-NAS)]
- Part-Level Graph Convolutional Network for Skeleton-Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6759)]
- Learning Diverse Stochastic Human-Action Generators by Learning Smooth Latent Transitions [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/6911)]

**ACM MM**
- Stronger, Faster and More Explainable: A Graph Convolutional Baseline for Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2010.09978.pdf)] [[code](https://gitee.com/yfsong0709/ResGCNv1)]
- Dynamic GCN: Context-enriched Topology Learning for Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2007.14690.pdf)] [[code](https://github.com/hikvision-research/skelact)]
- Spatio-Temporal Inception Graph Convolutional Networks for Skeleton-Based Action Recognition [[paper](https://arxiv.org/pdf/2011.13322.pdf)] [[code](https://github.com/yellowtownhz/STIGCN)]
- MS2L: Multi-Task Self-Supervised Learning for Skeleton Based Action Recognition [[paper](https://arxiv.org/pdf/2010.05599.pdf)] [[code](https://github.com/LanglandsLin/MS2L)]
- Action2Motion: Conditioned Generation of 3D Human Motions [[paper](https://arxiv.org/pdf/2007.15240.pdf)] [[code](https://github.com/EricGuo5513/action-to-motion)]
- Group-Skeleton-Based Human Action Recognition in Complex Events [[paper](https://arxiv.org/ftp/arxiv/papers/2011/2011.13273.pdf)]
- Mix Dimension in Poincaré Geometry for 3D Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2007.15678.pdf)]

**NIPSW**
- Contrastive Self-Supervised Learning for Skeleton Action Recognition [[paper](http://proceedings.mlr.press/v148/gao21a/gao21a.pdf)]

**ACCV**
- Decoupled Spatial-Temporal Attention Network for Skeleton-Based Action-Gesture Recognition [[paper](https://openaccess.thecvf.com/content/ACCV2020/papers/Shi_Decoupled_Spatial-Temporal_Attention_Network_for_Skeleton-Based_Action-Gesture_Recognition_ACCV_2020_paper.pdf)]

**TPAMI**
- Learning Multi-View Interactional Skeleton Graph for Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9234715)] [[code](https://github.com/niais/mv-ignet)]
- Multi-Task Deep Learning for Real-Time 3D Human Pose Estimation and Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9007695)] [[code](https://github.com/dluvizon/deephar)]

**TIP**
- Skeleton-Based Action Recognition with Multi-Stream Adaptive Graph Convolutional Networks [[paper](https://arxiv.org/pdf/1912.06971.pdf)] [[code](https://github.com/lshiwjx/2s-AGCN)]

**TMM**
- Hierarchical Soft Quantization for Skeleton-Based Human Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/9076822)]
- Deep Manifold-to-Manifold Transforming Network for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/8960323)]

**TCSVT**
- Richly Activated Graph Convolutional Network for Robust Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/2008.03791.pdf)] [[code](https://github.com/wqk666999/RA-GCNv2)]

**TNNLS**
- Adversarial Attack on Skeleton-Based Human Action Recognition [[paper](https://arxiv.org/pdf/1909.06500.pdf)]

**TOMM**
- A Benchmark Dataset and Comparison Study for Multi-modal Human Action Analytics [[paper](http://39.96.165.147/Pub%20Files/2020/ssj_tomm20.pdf)]

**PR**
- Skeleton-based action recognition with hierarchical spatial reasoning and temporal stack learning network [[paper](https://www.sciencedirect.com/science/article/pii/S0031320320303149)]

**Neurocomputing**
- Exploring a rich spatial–temporal dependent relational model for skeleton-based action recognition by bidirectional LSTM-CNN [[paper](https://www.sciencedirect.com/science/article/pii/S0925231220311760)]
- HDS-SP: A novel descriptor for skeleton-based human action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231219316509)]


### 2019

**CVPR**
- Two-Stream Adaptive Graph Convolutional Networks for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Two-Stream_Adaptive_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf)] [[code](https://github.com/lshiwjx/2s-AGCN)]
- Actional-Structural Graph Convolutional Networks for Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Li_Actional-Structural_Graph_Convolutional_Networks_for_Skeleton-Based_Action_Recognition_CVPR_2019_paper.pdf)] [[code](https://github.com/limaosen0/AS-GCN)]
- Skeleton-Based Action Recognition with Directed Graph Neural Networks [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Shi_Skeleton-Based_Action_Recognition_With_Directed_Graph_Neural_Networks_CVPR_2019_paper.pdf)] [[code](https://github.com/kenziyuliu/DGNN-PyTorch)]
- Bayesian Hierarchical Dynamic Model for Human Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhao_Bayesian_Hierarchical_Dynamic_Model_for_Human_Action_Recognition_CVPR_2019_paper.pdf)] [[code](https://github.com/rort1989/HDM)]
- An Attention Enhanced Graph Convolutional LSTM Network for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Si_An_Attention_Enhanced_Graph_Convolutional_LSTM_Network_for_Skeleton-Based_Action_CVPR_2019_paper.pdf)]

**ICCV**
- Bayesian Graph Convolution LSTM for Skeleton Based Action Recognition [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhao_Bayesian_Graph_Convolution_LSTM_for_Skeleton_Based_Action_Recognition_ICCV_2019_paper.pdf)]
- Making the Invisible Visible: Action Recognition Through Walls and Occlusions [[paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Making_the_Invisible_Visible_Action_Recognition_Through_Walls_and_Occlusions_ICCV_2019_paper.pdf)]

**AAAI**
- Graph CNNs with Motif and Variable Temporal Block for Skeleton-Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4929)] [[code](https://github.com/wenyh1616/motif-stgcn)]
- Spatio-Temporal Graph Routing for Skeleton-Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/4875)]

**CVPRW**
- Three-Stream Convolutional Neural Network With Multi-Task and Ensemble Learning for 3D Action Recognition [[paper](https://openaccess.thecvf.com/content_CVPRW_2019/papers/PBVS/Liang_Three-Stream_Convolutional_Neural_Network_With_Multi-Task_and_Ensemble_Learning_for_CVPRW_2019_paper.pdf)]

**ICCVW**
- Spatial Residual Layer and Dense Connection Block Enhanced Spatial Temporal Graph Convolutional Network for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content_ICCVW_2019/papers/SGRL/Wu_Spatial_Residual_Layer_and_Dense_Connection_Block_Enhanced_Spatial_Temporal_ICCVW_2019_paper.pdf)]

**WACV**
- Unsupervised Feature Learning of Human Actions As Trajectories in Pose Embedding Manifold [[paper](https://arxiv.org/abs/1812.02592)]

**ICIP**
- Richly Activated Graph Convolutional Network for Action Recognition with Incomplete Skeletons [[paper](https://arxiv.org/pdf/1905.06774.pdf)] [[code](https://gitee.com/yfsong0709/RA-GCNv1)]

**ICME**
- Skeleton-Based Action Recognition with Synchronous Local and Non-local Spatio-temporal Learning and Frequency Attention [[paper](https://arxiv.org/pdf/1811.04237.pdf)]
- Relational Network for Skeleton-Based Action Recognition [[paper](https://arxiv.org/pdf/1805.02556.pdf)]

**TPAMI**
- NTU RGB+D 120: A Large-Scale Benchmark for 3D Human Activity Understanding [[paper](https://arxiv.org/pdf/1905.04757.pdf)] [[code](https://github.com/shahroudy/NTURGB-D)]
- View Adaptive Neural Networks for High Performance Skeleton-based Human Action Recognition [[paper](https://arxiv.org/pdf/1804.07453.pdf)] [[code](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition)]

**TIP**
- Sample Fusion Network: An End-to-End Data Augmentation Network for Skeleton-Based Human Action Recognition [[paper](https://ieeexplore.ieee.org/document/8704987)] [[code](https://github.com/FanyangMeng/Sample-Fusion-Network)]
- View-Invariant Human Action Recognition Based on a 3D Bio-Constrained Skeleton Model [[paper](https://ieeexplore.ieee.org/abstract/document/8672922)] [[code](https://github.com/NIEQiang001/view-invariant-action-recognition-based-on-3D-bio-constrained-skeletons)]
- EleAtt-RNN: Adding Attentiveness to Neurons in Recurrent Neural Networks [[paper](https://arxiv.org/pdf/1909.01939.pdf)]
- Learning Latent Global Network for Skeleton-Based Action Prediction [[paper](https://ieeexplore.ieee.org/abstract/document/8822593)]

**TMM**
- 2-D Skeleton-Based Action Recognition via Two-Branch Stacked LSTM-RNNs [[paper](https://ieeexplore.ieee.org/abstract/document/8936339)]
- A Cuboid CNN Model With an Attention Mechanism for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/8943103)]
- Joint Learning in the Spatio-Temporal and Frequency Domains for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/8897586)]

**TCSVT**
- Action Recognition Scheme Based on Skeleton Representation With DS-LSTM Network [[paper](https://ieeexplore.ieee.org/abstract/document/8703407)]

**TNNLS**
- Graph Edge Convolutional Neural Networks for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/8842613)]

**Neurocomputing**
- Convolutional relation network for skeleton-based action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0925231219311816)]


### 2018

**CVPR**
- Recognizing Human Actions as the Evolution of Pose Estimation Maps [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Liu_Recognizing_Human_Actions_CVPR_2018_paper.pdf)] [[code](https://github.com/nkliuyifang/Skeleton-based-Human-Action-Recognition)]
- Independently Recurrent Neural Network (IndRNN): Building a Longer and Deeper RNN [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Independently_Recurrent_Neural_CVPR_2018_paper.pdf)] [[code](https://github.com/Sunnydreamrain/IndRNN_pytorch)]
- 2D/3D Pose Estimation and Action Recognition Using Multitask Deep Learning [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luvizon_2D3D_Pose_Estimation_CVPR_2018_paper.pdf)] [[code](https://github.com/dluvizon/deephar)]
- Deep Progressive Reinforcement Learning for Skeleton-Based Action Recognition [[paper](https://openaccess.thecvf.com/content_cvpr_2018/papers/Tang_Deep_Progressive_Reinforcement_CVPR_2018_paper.pdf)]

**ECCV**
- Skeleton-Based Action Recognition with Spatial Reasoning and Temporal Stack [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Chenyang_Si_Skeleton-Based_Action_Recognition_ECCV_2018_paper.pdf)]
- Adding Attentiveness to the Neurons in Recurrent Neural Networks [[paper](https://openaccess.thecvf.com/content_ECCV_2018/papers/Pengfei_Zhang_Adding_Attentiveness_to_ECCV_2018_paper.pdf)]

**AAAI**
- Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition [[paper](https://ojs.aaai.org/index.php/aaai/article/view/12328) [[code](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md)] [:fire:] [:star:]
- Unsupervised Representation Learning With Long-Term Dynamics for Skeleton Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11853)] [[code](https://github.com/jungel2star/Unsupervised-Representation-Learning-with-Long-Term-Dynamics-for-Skeleton-Based-Action-Recognition)]
- Spatio-Temporal Graph Convolution for Skeleton Based Action Recognition [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11776)]

**ACM MM**
- Optimized Skeleton-based Action Recognition via Sparsified Graph Regression [[paper](https://arxiv.org/pdf/1811.12013.pdf)]
- A Large-scale Varying-view RGB-D Action Dataset for Arbitrary-view Human Action Recognition [[paper](https://arxiv.org/pdf/1904.10681.pdf)]

**IJCAI**
- Co-occurrence Feature Learning from Skeleton Data for Action Recognition and Detection with Hierarchical Aggregation [[paper](https://arxiv.org/pdf/1804.06055.pdf)] [[code](https://github.com/huguyuehuhu/HCN-pytorch)]
- Memory Attention Networks for Skeleton-based Action Recognition [[paper](https://arxiv.org/pdf/1804.08254.pdf)] [[code](https://github.com/memory-attention-networks/MANs)]

**BMVC**
- Part-based Graph Convolutional Network for Action Recognition [[paper](https://arxiv.org/abs/1809.04983)] [[code](https://github.com/kalpitthakkar/pb-gcn)] 
- A Fine-to-Coarse Convolutional Neural Network for 3D Human Action Recognition [[paper](https://arxiv.org/abs/1805.11790)] 

**ICIP**
- Joints Relation Inference Network for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/8802912)]

**ICME**
- Skeleton-Based Human Action Recognition Using Spatial Temporal 3D Convolutional Neural Networks [[paper](https://ieeexplore.ieee.org/abstract/document/8486566)]

**TIP**
- Beyond Joints: Learning Representations From Primitive Geometries for Skeleton-Based Action Recognition and Detection [[paper](https://ieeexplore.ieee.org/abstract/document/8360391)] [[code](https://github.com/hongsong-wang/Beyond-Joints)]
- Learning Clip Representations for Skeleton-Based 3D Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/8306456)]

**TMM**
- Attention-Based Multiview Re-Observation Fusion Network for Skeletal Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/8421041)]
- Fusing Geometric Features for Skeleton-Based Action Recognition Using Multilayer LSTM Networks [[paper](https://ieeexplore.ieee.org/abstract/document/8281637)]

**TCSVT**
- Skeleton-Based Action Recognition With Gated Convolutional Neural Networks [[paper](https://ieeexplore.ieee.org/abstract/document/8529271)]
- Action Recognition With Spatio–Temporal Visual Attention on Skeleton Image Sequences [[paper](https://arxiv.org/pdf/1801.10304.pdf)]

**PR**
- Learning content and style: Joint action recognition and person identification from human skeletons [[paper](https://www.sciencedirect.com/science/article/pii/S0031320318301195)]


### 2017

**CVPR**
- Deep Learning on Lie Groups for Skeleton-based Action Recognition [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Huang_Deep_Learning_on_CVPR_2017_paper.pdf)] [[code](https://github.com/zhiwu-huang/LieNet)]
- Modeling Temporal Dynamics and Spatial Configurations of Actions Using Two-Stream Recurrent Neural Networks [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_Modeling_Temporal_Dynamics_CVPR_2017_paper.pdf)] 
- Global Context-Aware Attention LSTM Networks for 3D Action Recognition [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Liu_Global_Context-Aware_Attention_CVPR_2017_paper.pdf)]
- A New Representation of Skeleton Sequences for 3D Action Recognition [[paper](https://openaccess.thecvf.com/content_cvpr_2017/papers/Ke_A_New_Representation_CVPR_2017_paper.pdf)]

**ICCV**
- View Adaptive Recurrent Neural Networks for High Performance Human Action Recognition from Skeleton Data [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_View_Adaptive_Recurrent_ICCV_2017_paper.pdf)] [[code](https://github.com/microsoft/View-Adaptive-Neural-Networks-for-Skeleton-based-Human-Action-Recognition)]
- Ensemble Deep Learning for Skeleton-Based Action Recognition Using Temporal Sliding LSTM Networks [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lee_Ensemble_Deep_Learning_ICCV_2017_paper.pdf)] [[code](https://github.com/InwoongLee/TS-LSTM)]
- Learning Action Recognition Model From Depth and Skeleton Videos [[paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Rahmani_Learning_Action_Recognition_ICCV_2017_paper.pdf)]

**AAAI**
- An End-to-End Spatio-Temporal Attention Model for Human Action Recognition from Skeleton Data [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11212)]

**CVPRW**
- Interpretable 3D Human Action Analysis with Temporal Convolutional Networks [[paper](https://arxiv.org/pdf/1704.04516.pdf)] [[code](https://github.com/TaeSoo-Kim/TCNActionRecognition)]

**ICMEW**
- Skeleton based action recognition using translation-scale invariant image mapping and multi-scale deep CNN [[paper](https://arxiv.org/pdf/1704.05645.pdf)]
- Investigation of different skeleton features for CNN-based 3D action recognition [[paper](https://arxiv.org/pdf/1705.00835.pdf)]
- Skeleton-based action recognition using LSTM and CNN [[paper](https://arxiv.org/pdf/1707.02356.pdf)]

**TPAMI**
- Skeleton-Based Action Recognition Using Spatio-Temporal LSTM Network with Trust Gates [[paper](https://arxiv.org/pdf/1706.08276.pdf)] [[code](https://github.com/chungyin383/STLSTM)]

**TIP**
- Skeleton-Based Human Action Recognition With Global Context-Aware Attention LSTM Networks [[paper](https://arxiv.org/pdf/1707.05740.pdf)]

**PR**
- Learning discriminative trajectorylet detector sets for accurate skeleton-based action recognition [[paper](https://arxiv.org/pdf/1504.04923.pdf)]
- Enhanced skeleton visualization for view invariant human action recognition [[paper](https://www.sciencedirect.com/science/article/pii/S0031320317300936)]


### 2016

**CVPR**
- NTU RGB+D: A Large Scale Dataset for 3D Human Activity Analysis [[paper](https://openaccess.thecvf.com/content_cvpr_2016/papers/Shahroudy_NTU_RGBD_A_CVPR_2016_paper.pdf)] [[code](https://github.com/shahroudy/NTURGB-D)]
- Rolling Rotations for Recognizing Human Actions from 3D Skeletal Data [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Vemulapalli_Rolling_Rotations_for_CVPR_2016_paper.pdf)]

**ECCV**
- Temporal segment networks: Towards good practices for deep action recognition [[paper](https://arxiv.org/pdf/1608.00859.pdf%EF%BC%89)] [[code](https://github.com/yjxiong/temporal-segment-networks)]
- Spatio-Temporal LSTM with Trust Gates for 3D Human Action Recognition [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46487-9_50)]

**AAAI**
- Co-occurrence Feature Learning for Skeleton based Action Recognition using Regularized Deep LSTM Networks [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/10451)]

**ACM MM**
- Action Recognition Based on Joint Trajectory Maps Using Convolutional Neural Networks [[paper](https://arxiv.org/pdf/1611.02447.pdf)]

**TIP**
- Representation Learning of Temporal Dynamics for Skeleton-Based Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/7450165)]

**TMM**
- Discriminative Multi-instance Multitask Learning for 3D Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/7740059)]

**TCSVT**
- Skeleton Optical Spectra-Based Action Recognition Using Convolutional Neural Networks [[paper](https://ieeexplore.ieee.org/abstract/document/7742919)]


### 2015

**CVPR**
- Hierarchical Recurrent Neural Network for Skeleton Based Action Recognition [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Du_Hierarchical_Recurrent_Neural_2015_CVPR_paper.pdf)]
- Jointly learning heterogeneous features for RGB-D activity recognition [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Hu_Jointly_Learning_Heterogeneous_2015_CVPR_paper.pdf)]

**ICCV**
- Learning Spatiotemporal Features with 3D Convolutional Networks [[paper](https://openaccess.thecvf.com/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf)] [[code](https://vlg.cs.dartmouth.edu/c3d/)]

**TPAMI**
- Multimodal Multipart Learning for Action Recognition in Depth Videos [[paper](https://arxiv.org/pdf/1507.08761.pdf)] 

**TMM**
- Effective Active Skeleton Representation for Low Latency Human Action Recognition [[paper](https://ieeexplore.ieee.org/abstract/document/7346460)] 

**Neurocomputing**
- Skeleton-based action recognition with extreme learning machines [[paper](https://www.sciencedirect.com/science/article/pii/S0925231214011321)] 

### 2014

**CVPR**  
- Cross-view Action Modeling, Learning and Recognition [[paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Wang_Cross-view_Action_Modeling_2014_CVPR_paper.pdf)]
- Human Action Recognition by Representing 3D Skeletons as Points in a Lie Group [[paper](https://openaccess.thecvf.com/content_cvpr_2014/papers/Vemulapalli_Human_Action_Recognition_2014_CVPR_paper.pdf)]

**NeurIPS** 
- Two-Stream Convolutional Networks for Action Recognition in Videos [[paper](https://proceedings.neurips.cc/paper/2014/file/00ec53c4682d36f5c4359f4ae7bd7ba1-Paper.pdf)]

## Other Resources
With all the resources available on the github website, this paper list is comprehensive and recently updated.

- [niais/Awesome-Skeleton-based-Action-Recognition](https://github.com/niais/Awesome-Skeleton-based-Action-Recognition)
- [Kali-Hac/Awesome-Skeleton-Based-Models](https://github.com/Kali-Hac/Awesome-Skeleton-Based-Models)
- [qbxlvnf11/skeleton-based-action-recognition-methods](https://github.com/qbxlvnf11/skeleton-based-action-recognition-methods)
- [cagbal/Skeleton-Based-Action-Recognition-Papers-and-Notes](https://github.com/cagbal/Skeleton-Based-Action-Recognition-Papers-and-Notes)
- [XiaoCode-er/Skeleton-Based-Action-Recognition-Papers](https://github.com/XiaoCode-er/Skeleton-Based-Action-Recognition-Papers)
- [leviethung2103/awesome-skeleton-based-action-recognition](https://github.com/leviethung2103/awesome-skeleton-based-action-recognition)
- [fdu-wuyuan/Siren](https://github.com/fdu-wuyuan/Siren)
- [manjunath5496/Skeleton-based-Action-Recognition-Papers](https://github.com/manjunath5496/Skeleton-based-Action-Recognition-Papers)
- [liaomingg/action_recognition_and_skeleton_detection_summary](https://github.com/liaomingg/action_recognition_and_skeleton_detection_summary)
- [caglarmert/MOT-Research/wiki/Awesome-Action-Recognition](https://github.com/caglarmert/MOT-Research/wiki/Awesome-Action-Recognition)
- [shuangshuangguo/skeleton-based-action-recognition-review](https://github.com/shuangshuangguo/skeleton-based-action-recognition-review)

## Last update: Oct 26, 2023

## Feel free to contact me if you find any interesting paper is missing.
