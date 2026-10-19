# 数据来源与使用范围

T-LESS：Hodaň 等，T-LESS: An RGB-D Dataset for 6D Pose Estimation of Texture-less Objects，WACV 2017。原作者页面：https://cmp.felk.cvut.cz/t-less/ 。许可：CC BY 4.0。采用 models_cad 中人工创建的 CAD，原始压缩包同时保留重建与评价模型，后两者不替代本实验的 CAD 输入。

MVTec ITODD：Drost 等，Introducing MVTec ITODD – A Dataset for 3D Object Recognition in Industry，ICCV Workshops 2017。原作者页面：https://www.mvtec.com/research-teaching/datasets/mvtec-itodd 。原论文第 3 节明确说明全部 28 种对象提供人工 CAD。许可：CC BY-NC-SA 4.0，本实验用于非商业学术研究。

下载来自 BOP 官方 Hugging Face 数据仓库，版本与 SHA-256 位于 ../configs/dataset_sources.json。原始模型不改写。派生资产只作毫米到米的单位转换和确定性表面采样；之后的支撑坐标变换、碰撞近似、渲染材质和仿真物理属性须单独记录。

CAD 几何、原始真实照片与本项目合成双目图像属于不同的数据来源。仿真中指定的质量、摩擦和材料外观是实验设置，不作为原数据集提供的实测属性。
