import pandas as pd


class TreeLanguageVocab:
    def __init__(self, n_symbols, use_mask_token=False, use_cls_token=False):
        self.n_symbols = n_symbols
        self.special_tokens = []

        # Define the dictionary of "basic" symbols (with keys as strings),
        # mapping symbols (keys) to tokens (values).
        self.symbols_dict = {
            f'{i}': i
            for i in range(n_symbols)
        }

        # Add the <mask> symbol to the dictionary and to the list of special
        # tokens, if required.
        if use_mask_token:
            self.symbols_dict['<mask>'] = len(self.symbols_dict.values())
            self.special_tokens.append('<mask>')

        # Add the <cls> symbol to the dictionary and to the list of special
        # tokens, if required.
        if use_cls_token:
            self.symbols_dict['<cls>'] = len(self.symbols_dict.values())

            self.special_tokens.append('<cls>')

        self.df = pd.DataFrame([
            {'symbol': s, 'token': t}
            for s, t in self.symbols_dict.items()
        ])

        # Inverse relation to `simbols_dict`, mapping tokens (keys) to symbols
        # (values).
        self.tokens_dict = {
            r['token']: r['symbol']
            for i, r in self.df.sort_values(by='token').iterrows()
        }

    def __call__(self,  symbol):
        if str(symbol) not in self.symbols_dict.keys():
            raise Exception(f'Symbol {symbol} not found in vocabulary')
        else:
            return self.symbols_dict[str(symbol)]
        
    def __len__(self):
        """
        Returns the total size of the dictionary.
        """
        return len(self.symbols_dict)
    
    def sequence_to_tokens(self, sequence):
        """
        Given a sequence of symbols, returns the corresponding list of tokens.
        Special symbols must appear as strings ('<mask>', '<cls>', ...).
        """
        if all([isinstance(s, str) for s in sequence]):
            return [
                self(symbol)
                for symbol in sequence
            ]
        else:
            return [
                self(str(symbol))
                for symbol in sequence
            ]
        
    def tokens_to_sequence(self, tokens):
        """
        Given a sequence of tokens, returns the corresponding list of symbols.
        """
        return [
            self.tokens_dict[t]
            for t in tokens
        ]
