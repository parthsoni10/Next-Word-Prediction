import h5py
import pickle
import numpy as np
import os

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except Exception:
            class GenericClass:
                def __init__(self, *args, **kwargs): pass
                def __setstate__(self, state): self.__dict__.update(state)
            return GenericClass

class NextWordPredictor:
    def __init__(self, model_path='lstm_model.h5', tokenizer_path='tokenizer.pkl', max_len_path='max_len.pkl'):
        # 1. Load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tok_obj = CustomUnpickler(f).load()
        
        self.word_index = getattr(tok_obj, 'word_index', {})
        self.index_word = getattr(tok_obj, 'index_word', {})
        # Ensure integer keys in index_word map
        self.index_word = {int(k): str(v) for k, v in self.index_word.items()}
        self.filters = getattr(tok_obj, 'filters', '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.lower = getattr(tok_obj, 'lower', True)

        # 2. Load max_len
        with open(max_len_path, 'rb') as f:
            self.max_len = pickle.load(f)

        # 3. Load weights directly from HDF5
        with h5py.File(model_path, 'r') as f:
            mw = f['model_weights']
            self.embeddings = np.array(mw['embedding_1/sequential_1/embedding_1/embeddings'])
            
            lstm_grp = mw['lstm/sequential_1/lstm/lstm_cell']
            self.lstm_kernel = np.array(lstm_grp['kernel'])            # (50, 512)
            self.lstm_recurrent_kernel = np.array(lstm_grp['recurrent_kernel']) # (128, 512)
            self.lstm_bias = np.array(lstm_grp['bias'])                # (512,)
            
            dense_grp = mw['dense_1/sequential_1/dense_1']
            self.dense_kernel = np.array(dense_grp['kernel'])          # (128, 10000)
            self.dense_bias = np.array(dense_grp['bias'])              # (10000,)

        self.vocab_size = len(self.word_index)
        self.units = 128
        self.embed_dim = self.embeddings.shape[1]

    def clean_text(self, text):
        if self.lower:
            text = text.lower()
        translate_dict = str.maketrans('', '', self.filters)
        text = text.translate(translate_dict)
        return text.split()

    def text_to_sequence(self, text):
        words = self.clean_text(text)
        seq = [self.word_index.get(w, 1) for w in words] # 1 is OOV token if unknown
        return seq

    def pad_sequence(self, seq):
        target_len = self.max_len - 1 if self.max_len > 1 else self.max_len
        if len(seq) > target_len:
            seq = seq[-target_len:]
        elif len(seq) < target_len:
            seq = [0] * (target_len - len(seq)) + seq
        return seq

    def forward_lstm(self, X_seq):
        h = np.zeros(self.units, dtype=np.float32)
        c = np.zeros(self.units, dtype=np.float32)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

        for token_id in X_seq:
            if token_id <= 0 or token_id >= self.embeddings.shape[0]:
                x_t = np.zeros(self.embed_dim, dtype=np.float32)
            else:
                x_t = self.embeddings[token_id]

            gates = np.dot(x_t, self.lstm_kernel) + np.dot(h, self.lstm_recurrent_kernel) + self.lstm_bias
            
            i = sigmoid(gates[0:128])
            f = sigmoid(gates[128:256])
            c_tilde = np.tanh(gates[256:384])
            o = sigmoid(gates[384:512])

            c = f * c + i * c_tilde
            h = o * np.tanh(c)

        return h

    def predict_next_words(self, text, top_k=5):
        seq = self.text_to_sequence(text)
        if not seq:
            return []
        
        padded = self.pad_sequence(seq)
        h_last = self.forward_lstm(padded)
        
        logits = np.dot(h_last, self.dense_kernel) + self.dense_bias
        
        # Softmax computation
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        
        top_indices = np.argsort(probs)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            word = self.index_word.get(idx, '<UNKNOWN>')
            prob = float(probs[idx])
            results.append({"word": word, "probability": prob, "token_id": int(idx)})
            
        return results

    def generate_text(self, seed_text, num_words=10, temperature=1.0):
        current_text = seed_text
        generated_words = []
        
        for _ in range(num_words):
            seq = self.text_to_sequence(current_text)
            if not seq:
                break
            padded = self.pad_sequence(seq)
            h_last = self.forward_lstm(padded)
            logits = np.dot(h_last, self.dense_kernel) + self.dense_bias
            
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature
            
            exp_logits = np.exp(logits - np.max(logits))
            probs = exp_logits / np.sum(exp_logits)
            
            next_idx = np.random.choice(len(probs), p=probs)
            next_word = self.index_word.get(next_idx, '')
            if not next_word or next_word == '<OOV>':
                top_preds = self.predict_next_words(current_text, top_k=1)
                if top_preds:
                    next_word = top_preds[0]['word']
                else:
                    break
            generated_words.append(next_word)
            current_text += " " + next_word

        return current_text, generated_words
