import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from nltk.translate import bleu_score
import nltk
nltk.download('punkt')

# Load pre-trained ResNet model
resnet = models.resnet152(pretrained=True)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.eval()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load word tokenizer
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
vocab = torch.load('vocab.pth')  # Pre-processed vocabulary

# Define the captioning model
class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, features, captions, lengths):
        embeddings = self.embedding(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True)
        hiddens, _ = self.lstm(packed_embeddings)
        hiddens, _ = nn.utils.rnn.pad_packed_sequence(hiddens, batch_first=True)
        attention_weights = self.attention(hiddens, features)
        context = attention_weights.bmm(features)
        outputs = self.fc(context + hiddens)
        return outputs

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        
    def forward(self, hiddens, features):
        score = torch.tanh(self.W1(hiddens) + self.W2(features.unsqueeze(1)))
        attention_weights = torch.softmax(self.V(score), dim=1)
        return attention_weights

# Load captioning model
captioning_model = CaptioningModel(embed_size=256, hidden_size=512, vocab_size=len(vocab))

# Load pre-trained captioning model weights
captioning_model.load_state_dict(torch.load('captioning_model.pth'))
captioning_model.eval()

# Load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

# Generate a caption for an image
def generate_caption(image):
    with torch.no_grad():
        features = resnet(image)
        features = features.view(features.size(0), -1)
        sampled_ids = []
        inputs = torch.tensor([vocab['<start>']])
        h, c = None, None

        for _ in range(20):  # Maximum sequence length
            embeddings = captioning_model.embedding(inputs)
            if h is not None:
                hiddens, (h, c) = captioning_model.lstm(embeddings.unsqueeze(1), (h, c))
            else:
                hiddens, (h, c) = captioning_model.lstm(embeddings.unsqueeze(1))
                
            attention_weights = captioning_model.attention(hiddens, features)
            context = attention_weights.bmm(features.unsqueeze(0))
            output = captioning_model.fc(context + hiddens)
            
            _, predicted = output.max(2)
            sampled_ids.append(predicted.item())
            
            if predicted.item() == vocab['<end>']:
                break
            
            inputs = predicted.squeeze()

        caption = [vocab.idx2word[word_id] for word_id in sampled_ids]
        caption = ' '.join(caption[1:-1])  # Remove <start> and <end> tokens
        return caption

# Load an image and generate a caption
image_path = 'path/to/your/image.jpg'
image = load_image(image_path)
caption = generate_caption(image)
print("Generated Caption:", caption)
