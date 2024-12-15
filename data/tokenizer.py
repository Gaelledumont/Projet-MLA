import sentencepiece as spm

class CamemBertTokenizer:
    def __init__(self, sp_model_path, special_tokens=["<s>", "</s>", "<unk>", "<pad>", "<mask>"]):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self.cls_token = "<s>"
        self.sep_token = "</s>"
        self.unk_token = "<unk>"
        self.pad_token = "<pad>"
        self.mask_token = "<mask>"

    def encode(self, text, max_length=512, add_special_tokens=True):
        tokens = self.sp.EncodeAsPieces(text)
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        ids = [self.sp.PieceToId(t) for t in tokens]
        if len(ids) > max_length:
            ids = ids[:max_length-1] + [self.sp.PieceToId(self.pad_token)]
        return ids

    def get_vocab_size(self):
        return self.sp.GetVocabSize()

    def mask_token_id(self):
        return self.sp.PieceToId(self.mask_token)

    def pad_token_id(self):
        return self.sp.PieceToId(self.pad_token)

    def cls_token_id(self):
        return self.sp.PieceToId(self.cls_token)

    def sep_token_id(self):
        return self.sp.PieceToId(self.sep_token)

    def unk_token_id(self):
        return self.sp.PieceToId(self.unk_token)