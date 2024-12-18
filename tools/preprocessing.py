import torch
from transformers import BertTokenizer, BertModel, ViTImageProcessor, ViTModel
from PIL import Image
import os


class TextProcessor:
    def __init__(self, model_path, device='cpu', max_length=24):
        """初始化BERT模型和分词器"""
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path).to(device)
        self.device = device
        self.max_length = max_length

    def process(self, sentences):
        """处理文本，获取统一长度的BERT特征"""
        self.model.eval()
        all_embeddings = []
        with torch.no_grad():
            for sentence in sentences:
                inputs = self.tokenizer(sentence, return_tensors="pt", padding="max_length", truncation=True, max_length=self.max_length)
                input_ids = inputs['input_ids'].to(self.device)
                attention_mask = inputs['attention_mask'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask)
                embeddings = outputs.last_hidden_state  # [batch_size, max_length, hidden_dim]

                # 保证输出的特征长度为 max_length
                if embeddings.size(1) < self.max_length:
                    padding = torch.zeros(embeddings.size(0), self.max_length - embeddings.size(1), embeddings.size(2)).to(self.device)
                    embeddings = torch.cat([embeddings, padding], dim=1)
                else:
                    embeddings = embeddings[:, :self.max_length, :]

                all_embeddings.append(embeddings.cpu())
        return all_embeddings

class ImageProcessor:
    def __init__(self, model_path):
        """初始化ViT处理器和模型"""
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTModel.from_pretrained(model_path)

    def process_image(self, image_path):
        """处理单张图片"""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.last_hidden_state  # 返回图像特征张量

    def process_images(self, folder_path):
        """批量处理图片"""
        pixel_values = []
        for file_name in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file_name)
            if image_path.lower().endswith(('.png', 'jpg', 'jpeg')):
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    pixel_values.append(inputs["pixel_values"])
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
        if pixel_values:
            # 检查所有图片是否具有相同的尺寸
            max_height, max_width = 224, 224  # ViT 默认图像大小
            for i, tensor in enumerate(pixel_values):
                if tensor.size(-2) != max_height or tensor.size(-1) != max_width:
                    raise ValueError(f"Inconsistent image sizes detected in batch. File index: {i}")
            return torch.cat(pixel_values, dim=0)  # (batch_size, 3, height, width)
        else:
            raise ValueError(f"No valid images found in {folder_path}")
        
class DataLoader:
    @staticmethod
    def load_data(file_path):
        """加载文本数据"""
        sentences, labels, img_ids = [], [], []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for i in range(0, len(lines), 4):
            if i + 3 >= len(lines):
                print(f"Skipping incomplete record at line {i}")
                break
            sentence_1 = lines[i].strip()
            sentence_2 = lines[i + 1].strip()
            label = int(lines[i + 2].strip())
            img_id = lines[i + 3].strip()
            full_sentence = f"{sentence_1} {sentence_2}"
            sentences.append(full_sentence)
            labels.append(label)
            img_ids.append(img_id)
        return sentences, labels, img_ids
    
    def save_preprocessed_data(self, text_features, image_features, labels, save_dir):
        """
        保存预处理结果
        :param text_features: 文本特征列表 [num_samples, seq_len, feature_dim]
        :param image_features: 图像特征张量 [num_samples, seq_len, feature_dim]
        :param labels: 标签列表 [num_samples]
        :param save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存文本特征
        torch.save(text_features, os.path.join(save_dir, "text_features.pt"))
        # 保存图像特征
        torch.save(image_features, os.path.join(save_dir, "image_features.pt"))
        # 保存标签
        torch.save(labels, os.path.join(save_dir, "labels.pt"))

        print(f"Data saved in {save_dir}")

    
    def load_preprocessed_data(self, save_dir):
        """
        加载预处理结果
        :param save_dir: 保存目录
        :return: 文本特征, 图像特征, 标签
        """
        text_features = torch.load(os.path.join(save_dir, "text_features.pt"))
        image_features = torch.load(os.path.join(save_dir, "image_features.pt"))
        labels = torch.load(os.path.join(save_dir, "labels.pt"))

        print("Data loaded successfully")
        return text_features, image_features, labels


def save_features(text_features, image_features, save_path):
    """保存特征为pt文件"""
    # 确保父目录存在
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    # 保存特征
    torch.save({'text': text_features, 'image': image_features}, save_path)
    print(f"Features saved to {save_path}")

if __name__ == "__main__":
    BERT_PATH = './modules/models/BERT'
    VIT_PATH = './modules/models/Vit'
    IMAGE_FOLDER = './img_data/twitter2015'
    TRAIN_PATH = "./src_data/data_baseline/twitter2015/train.txt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 加载数据
    data_loader = DataLoader()
    sentences, labels, img_ids = data_loader.load_data(TRAIN_PATH)
    print(f"Number of records: {len(img_ids)}")
    print(f"Example: {sentences[0]}, Label: {labels[0]}, IMGID: {img_ids[0]}")

    # 处理文本
    text_processor = TextProcessor(BERT_PATH, device=DEVICE)
    sentence_embeddings = text_processor.process(sentences)
    print("Text embeddings shape:", sentence_embeddings[0].shape)

    # 处理图片
    image_processor = ImageProcessor(VIT_PATH)
    batch_inputs = image_processor.process_images(IMAGE_FOLDER)
    print("Image batch shape:", batch_inputs.shape)
    text_features = torch.cat(text_processor.process(sentences)).to(DEVICE)
    image_features = image_processor.process_images(IMAGE_FOLDER).to(DEVICE)

    save_features(text_features, image_features, '../src_data/preprocessed_data/features.pt')