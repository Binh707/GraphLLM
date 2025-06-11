import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from transformers import BertTokenizer, BertModel
from transformers import DistilBertModel, DistilBertTokenizer
import torch.nn.functional as F
from transformers import GPT2Model, GPT2Tokenizer
from tqdm import tqdm
from peft import get_peft_model, LoraConfig
from transformers import GPT2Tokenizer, GPT2Model




class GraphTextModel(nn.Module):
    def __init__(self, feature_dim, text_embedding_dim, num_classes,texts,embedding_dim=128, num_gcn_layers=2, Lora=True, soft=True,LM='BERT'):
        super(GraphTextModel, self).__init__()
        if LM=="GPT":
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.text_model = GPT2Model.from_pretrained('gpt2')
        if LM=="BERT":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.text_model = BertModel.from_pretrained("bert-base-uncased")
        self.texts=texts
        self.soft=soft
        self.ML=LM
            
        for param in self.text_model.parameters():
            param.requires_grad = False
        if Lora and LM=="BERT":
            lora_config = LoraConfig(
                r=8,  # اندازه رنک LoRA
                lora_alpha=32,  # ضریب LoRA
                target_modules=["attention.self.query", "attention.self.key", "attention.self.value"],  # لایه‌هایی که باید LoRA روی آنها اعمال بشه
                lora_dropout=0.1,  # میزان دراپوت
                bias="none"  # می‌تونی bias رو به "all" تغییر بدی
            )
            # اضافه کردن LoRA به مدل BERT
            self.text_model = get_peft_model(self.text_model, lora_config)
        if Lora and LM=="GPT":
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["c_attn"],  # اینجا باید فقط "c_attn" باشه
                lora_dropout=0.1,
                bias="none"
            )
            self.text_model = get_peft_model(self.text_model, lora_config)

        
        self.feature_transform = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, text_embedding_dim)
        )
        
        # self.gcn_layers = nn.ModuleList([
        #     GCNConv(text_embedding_dim, text_embedding_dim) 
        #     for _ in range(num_gcn_layers)
        # ])
        #-------- GAT layers
        # self.gcn_layers = nn.ModuleList([  
        #     GATConv(text_embedding_dim, text_embedding_dim)   
        #     for _ in range(num_gcn_layers)  
        # ])
        # num_heads = 8  # به عنوان مثال  
        # self.gcn_layers = nn.ModuleList([  
        #     GATConv(text_embedding_dim, text_embedding_dim // num_heads, heads=num_heads)  
        #     for _ in range(num_gcn_layers)  
        # ])
        #------------------- graph_SAGE     agg=mean
        self.gcn_layers = nn.ModuleList([  
            SAGEConv(text_embedding_dim, text_embedding_dim)   
            for _ in range(num_gcn_layers)  
        ])
        
        self.classifier = nn.Sequential(
            nn.Linear(text_embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        self.cross_attention = nn.MultiheadAttention(text_embedding_dim, 4)

    def forward(self, edge_index, n_id, feature_vec):
        transformed_feature = self.feature_transform(feature_vec)

        text = [self.texts[i] for i in n_id.cpu().numpy()]
        if self.ML=="BERT":
            tokens = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')
        elif self.ML=="GPT":
            self.tokenizer.pad_token = self.tokenizer.eos_token
            tokens = self.tokenizer(text, padding=True, truncation=True, max_length=30, return_tensors='pt')

        tokens = tokens.to(edge_index.device)
        input_embeddings = self.text_model.get_input_embeddings()(tokens['input_ids'])
        
        if self.soft == False:
            ## w\o soft prompt------------
            outputs = self.text_model(inputs_embeds=input_embeddings)
            hidden_states = outputs.last_hidden_state
            text_embedding = hidden_states[:, 0, :]
            ## w\o soft prompt------------

        graph_embedding = transformed_feature
        count = 0
        
        for gcn_layer in self.gcn_layers:
            edge_index = edge_index.long()

            
            if self.soft:
                ## w\ soft  prompt
                # گراف روی متن توجه می‌کند (Cross Attention)
                graph_embedding = graph_embedding.unsqueeze(1)
                modified_embeddings = torch.cat((graph_embedding, input_embeddings), dim=1)
                attention_mask = tokens['attention_mask']
                batch_size = attention_mask.shape[0]
                new_token_mask = torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([new_token_mask, attention_mask], dim=1)
                outputs = self.text_model(inputs_embeds=modified_embeddings, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state
                text_embedding = hidden_states[:, 0, :]
                ## w\ soft  prompt

            # متن روی گراف توجه می‌کند (اضافه شده)
            graph_embedding_for_attention = graph_embedding.squeeze(1).unsqueeze(0)  # [1, batch_size, embedding_dim]
            text_embedding_for_attention = text_embedding.unsqueeze(0)  # [1, batch_size, embedding_dim]
            text_to_graph_attention, _ = self.cross_attention(graph_embedding_for_attention, text_embedding_for_attention,text_embedding_for_attention)
            text_to_graph_attention = text_to_graph_attention.squeeze(0)  # [batch_size, embedding_dim]

            # ترکیب توجه گراف → متن و متن → گراف
            combined_embedding = (text_embedding + text_to_graph_attention) / 2
            # به‌روزرسانی گراف با GCN
            graph_embedding = gcn_layer(combined_embedding, edge_index)

        return self.classifier(graph_embedding),graph_embedding



class GNN(nn.Module):
    def __init__(self, feature_dim, GNN_name, num_classes, num_gcn_layers=2):
        super(GNN, self).__init__()
        if GNN_name=='GCN':
            self.gcn_layers = nn.ModuleList([
                GCNConv(feature_dim, feature_dim) 
                for _ in range(num_gcn_layers)
            ])
        elif GNN_name=='GAT':
            #-------- GAT layers
            self.gcn_layers = nn.ModuleList([  
                GATConv(feature_dim, feature_dim)   
                for _ in range(num_gcn_layers)  
            ])
        elif GNN_name=='SAGE':
            #------------------- graph_SAGE     agg=mean
            self.gcn_layers = nn.ModuleList([  
                SAGEConv(feature_dim, feature_dim)   
                for _ in range(num_gcn_layers)  
            ])
        
        
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, num_classes),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )


    def forward(self, edge_index, n_id, feature_vec):
        
        graph_embedding=feature_vec
       
        for gcn_layer in self.gcn_layers:
            edge_index = edge_index.long()
            # به‌روزرسانی گراف با GCN
            graph_embedding = gcn_layer(graph_embedding, edge_index)

        return self.classifier(graph_embedding),graph_embedding

class MLP(torch.nn.Module):
    def __init__(self, feature_dim,embedding_dim, num_layers,num_classes, dropout=0.2):
        super(MLP, self).__init__()
        # self.use_pred = use_pred
        # if self.use_pred:
        #     self.encoder = torch.nn.Embedding(embedding_dim+1, embedding_dim)
        self.convs = torch.nn.ModuleList()
        self.convs.append(nn.Linear(feature_dim, embedding_dim))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(embedding_dim))
        for _ in range(num_layers - 2):
            self.convs.append(nn.Linear(embedding_dim, embedding_dim))
            self.bns.append(torch.nn.BatchNorm1d(embedding_dim))
        self.convs.append(nn.Linear(embedding_dim, num_classes))

        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
   
    def forward(self, x, edge_index,n_id, feature_vec):
        # if self.use_pred:
        #     x = self.encoder(x)
        #     x = torch.flatten(x, start_dim=1)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x)

        return x, x
        
class TextGraphModel(nn.Module):
    def __init__(self, embedding_dim, num_classes, texts,num_gcn_layers=2, use_transformer_layers=True):
        super(TextGraphModel, self).__init__()
        
        # Load DistilGPT-2 model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
        self.gpt2_model = GPT2Model.from_pretrained('distilgpt2')
        
        # Freeze the language model parameters to avoid fine-tuning
        for param in self.gpt2_model.parameters():
            param.requires_grad = False
        
        # DistilGPT-2 hidden size (768)
        self.gpt2_hidden_size = self.gpt2_model.config.hidden_size
        print("self.gpt2_hidden_size=",self.gpt2_hidden_size)
        
        # Number of transformer layers in DistilGPT-2 (6 layers)
        self.num_layers = self.gpt2_model.config.n_layer
        print("self.num_layers=",self.num_layers)
        
        # Whether to use outputs from all transformer layers
        self.use_transformer_layers = use_transformer_layers
        
        # Separate projection layer for each transformer layer
        # self.layer_projections = nn.ModuleList([
        #     nn.Linear(self.gpt2_hidden_size, embedding_dim) 
        #     for _ in range(self.num_layers + 1)  # +1 for embedding layer
        # ])
        
        self.projection2 = nn.Sequential(
            nn.Linear(self.gpt2_hidden_size, embedding_dim),
            # nn.ReLU(),
            # nn.Linear(128, embedding_dim)
        )
        # self.gcn_layers = nn.ModuleList([
        #     GCNConv(text_embedding_dim, text_embedding_dim) 
        #     for _ in range(num_gcn_layers)
        # ])
        #-------- GAT layers
        # self.gcn_layers = nn.ModuleList([  
        #     GATConv(text_embedding_dim, text_embedding_dim)   
        #     for _ in range(num_gcn_layers)  
        # ])
        # num_heads = 8  # به عنوان مثال  
        # self.gcn_layers = nn.ModuleList([  
        #     GATConv(text_embedding_dim, text_embedding_dim // num_heads, heads=num_heads)  
        #     for _ in range(num_gcn_layers)  
        # ])
        #------------------- graph_SAGE     agg=mean
        # self.gcn_layers = nn.ModuleList([  
        #     SAGEConv(text_embedding_dim, text_embedding_dim)   
        #     for _ in range(num_gcn_layers)  
        # ])
        
        
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, num_classes),
            # nn.ReLU(),
            # nn.Linear(64, num_classes)
        )
        self.texts=texts
    
    def extract_text_features(self, texts):
        """
        Extract features from text using DistilGPT-2 model.
        Returns separate features for each transformer layer.
        
        Args:
            texts: List of strings, one for each node
            device: Device to run the model on
        Returns:
            List of torch.Tensor: List of text features for each layer
        """
        # Initialize list of tensors for each layer
        all_layer_features = [[] for _ in range(self.num_layers + 1)]
        
        for text in tqdm(texts, desc="Extracting text features"):
            # Tokenize text and move to device
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            # Get the position of the last token
            last_token_idx = torch.sum(inputs['attention_mask'], dim=1) - 1
            
            # Run through DistilGPT-2 with output_hidden_states=True to get all layer outputs
            with torch.no_grad():
                outputs = self.gpt2_model(**inputs, output_hidden_states=True)
            
            # Get the last token representation from each transformer layer separately
            for layer_idx in range(len(outputs.hidden_states)):
                layer_output = outputs.hidden_states[layer_idx]
                # Extract the last token representation
                last_token_feature = layer_output[0, last_token_idx]
                all_layer_features[layer_idx].append(last_token_feature.cpu())
        
        # Stack node text features for each layer
        return [torch.stack(layer_features) for layer_features in all_layer_features]
    
    def forward(self, x, edge_index,n_id, feature_vec):
        

        text = [self.texts[i] for i in n_id.cpu().numpy()]
        layer_text_features=self.extract_text_features(text)
        print("layer_text_features",layer_text_features.shape)

        layer_text_features=self.projection2(layer_text_features)
        print("layer_text_features=",layer_text_features)
        """
        Forward pass through the model.
        
        Args:
            edge_index: Edge indices of the graph
            layer_text_features: List of text features for each layer
            
        Returns:
            logits: Classification results for each layer
            node_embeddings: Node embeddings from GCN for each layer
        """
        if self.gcn_layers==1:
            index=[1,6]
        elif self.gcn_layers==2:
            index=[1,3,6]
        elif self.gcn_layers==3:
            index=[1,2,4,6]
        current_index=0
        for gcn_layer in self.gcn_layers:
            
            edge_index = edge_index.long()

            graph_embedding = gcn_layer(layer_text_features[index[current_index]], edge_index)
            current_index += 1
            # print("graph_embedding.shape",graph_embedding.shape)

        graph_embedding=(graph_embedding+layer_text_features[current_index])/2

        return self.classifier(graph_embedding),graph_embedding
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMLP(nn.Module):
    def __init__(self, feature_dim,embedding_dim, num_layers,num_classes, dropout=0.2):
        super(SimpleMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(feature_dim, embedding_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(embedding_dim, embedding_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(embedding_dim, num_classes))  # لایه خروجی

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, edge_index,n_id, feature_vec):
        return self.mlp(x),x
