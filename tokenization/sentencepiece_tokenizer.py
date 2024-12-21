import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, spm_model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

    def encode(self, text):
        return self.sp.encode_as_ids(text)

    def decode(self, token_ids):
        return self.sp.decode_ids(token_ids)

    def mask_token_id(self):
        return self.sp.PieceToId("<mask>")

    def pad_token_id(self):
        return 1

    def __len__(self):
        return self.sp.get_piece_size()