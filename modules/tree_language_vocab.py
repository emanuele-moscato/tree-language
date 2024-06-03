import pandas as pd


class TreeLanguageVocab:
    def __init__(
            self,
            n_symbols,
            use_mask_token=False,
            use_cls_token=False,
            use_pad_token=False
        ):
        self.n_symbols = n_symbols
        self.special_symbols = []
        self.use_mask_token = use_mask_token,
        self.use_cls_token = use_cls_token
        self.use_pad_token = use_pad_token

        # Define the dictionary of "basic" symbols (with keys as strings),
        # mapping symbols (keys) to tokens (values).
        self.symbols_dict = {
            f'{i}': i
            for i in range(n_symbols)
        }

        self.df = pd.DataFrame([
            {'symbol': s, 'token': t}
            for s, t in self.symbols_dict.items()
        ])

        # Inverse relation to `simbols_dict`, mapping tokens (keys) to symbols
        # (values).
        self.tokens_dict = {
            r['token']: r['symbol']
            for _, r in self.df.sort_values(by='token').iterrows()
        }

        # Add the <mask> symbol to the dictionary and to the list of special
        # tokens, if required.
        if use_mask_token:
            self.add_symbol('<mask>')
            self.special_symbols.append('<mask>')

        # Add the <cls> symbol to the dictionary and to the list of special
        # tokens, if required.
        if use_cls_token:
            self.add_symbol('<cls>')
            self.special_symbols.append('<cls>')

        # Add the <pad> symbol to the dictionary and to the list of special
        # tokens, if required.
        if use_pad_token:
            self.add_symbol('<pad>')
            self.special_symbols.append('<pad>')

    def add_symbol(self, new_symbol, special_symbol=False):
        """
        Adds the input symbol to the vocabulary. The corresponding token is
        `[max token] + 1`.
        """
        # If the new symbol already exists in the vocabulary, raise an
        # exception.
        if new_symbol in self.symbols_dict.keys():
            raise Exception(
                f'Symbol {new_symbol} already exists in the vocabulary'
            )

        new_token = len(self.symbols_dict.values())

        # Add (new_symbol, new_token) pair to the symbols dict, the vocab's
        # dataframe and the tokens dict.
        self.symbols_dict[str(new_symbol)] = new_token
        self.df = pd.concat([
            self.df,
            pd.DataFrame([{'symbol': new_symbol,'token': new_token}])
        ])
        self.tokens_dict[new_token] = new_symbol

        # If the new symbol must is considered among the special ones, add it
        # to the list of special symbols.
        if special_symbol:
            self.special_symbols.append(new_symbol)

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
