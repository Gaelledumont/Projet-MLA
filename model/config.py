class TransformerConfig:
    def __init__(self, vocab_size=32000, hidden_size=768, num_hidden_layers=12,
                 num_attention_heads=12, intermediate_size=3072, dropout_prob=0.1,
                 max_position_embeddings=514, layer_norm_eps=1e-5, pad_token_id=1,
                 hidden_act="gelu"):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.hidden_act = hidden_act