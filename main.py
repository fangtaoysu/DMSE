from tools.preprocessing import TextProcessor, ImageProcessor, DataLoader
from modules.fusion import MultiGranularityFusion
import torch
import os



    
def pad_to_max_length(embeddings, max_length):
    """
    对 BERT 输出的序列嵌入进行 padding
    :param embeddings: List of tensors, shape: [(seq_len, embedding_dim), ...]
    :param max_length: 统一的序列长度
    :return: Padded tensor, shape: [num_sentences, max_length, embedding_dim]
    """
    padded_embeddings = []
    for emb in embeddings:
        if emb.size(1) < max_length:
            # 填充到最大长度
            padding = torch.zeros(emb.size(0), max_length - emb.size(1), emb.size(2)).to(emb.device)
            padded_embeddings.append(torch.cat([emb, padding], dim=1))
        else:
            # 截断到最大长度
            padded_embeddings.append(emb[:, :max_length])
    return torch.stack(padded_embeddings)


def load_features(file_path):
    """加载pt文件中的特征"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    features = torch.load(file_path, weights_only=True)
    return features['text'], features['image']


if __name__ == "__main__":
    # 初始化路径
    text_features, image_features = load_features('./src_data/preprocessed_data/features.pt')
    print("Loaded text features:", text_features.shape)
    print("Loaded image features:", image_features.shape)

    # TODO 测试多模态融合的代码
    # # 加载数据
    # data_loader = DataLoader()
    # sentences, labels, img_ids = data_loader.load_data(TRAIN_PATH)
    # print(f"Number of records: {len(img_ids)}")
    # print(f"Example: {sentences[0]}, Label: {labels[0]}, IMGID: {img_ids[0]}")

    # # 处理文本
    # text_processor = TextProcessor(BERT_PATH, device=DEVICE)
    # sentence_embeddings = text_processor.process(sentences)
    # print("Text embeddings shape:", sentence_embeddings[0].shape)

    # # 处理图片
    # image_processor = ImageProcessor(VIT_PATH)
    # batch_inputs = image_processor.process_images(IMAGE_FOLDER)
    # print("Image batch shape:", batch_inputs.shape)

    # # 初始化模型参数
    # TEXT_DIM = 1024  # BERT 输出的维度
    # IMAGE_DIM = 1024  # ViT 输出的维度
    # FUSION_DIM = 512  # 融合特征维度

    # # 加载预处理数据
    # text_processor = TextProcessor(BERT_PATH, device=DEVICE)
    # image_processor = ImageProcessor(VIT_PATH)
    # sentences, _, _ = DataLoader.load_data(TRAIN_PATH)
    # sentence_embeddings = text_processor.process(sentences)
    # image_embeddings = image_processor.process_images(IMAGE_FOLDER)

    # # 初始化多模态融合模型
    # fusion_model = MultiGranularityFusion(TEXT_DIM, IMAGE_DIM, FUSION_DIM).to(DEVICE)

    # # 输入到融合模型
    # text_features = torch.cat(sentence_embeddings).to(DEVICE)  # [batch_size, seq_len, text_dim]
    # image_features = image_embeddings.to(DEVICE)  # [batch_size, seq_len, image_dim]

    # # 融合结果
    # fused_features = fusion_model(text_features, image_features)
    # print("Fused features shape:", fused_features.shape)  # [batch_size, seq_len, fusion_dim]