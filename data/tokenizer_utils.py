from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        result=[] 
        for num in numbers:
            s=str(num)
            tokens=[] 
            i=0 
            n=len(s)
            while i<n:
                match=False
                for j in range(n,i,-1):
                    sub=s[i:j]
                    if sub in vocab:
                        tokens.append(sub)
                        i=j 
                        match=True 
                        break 
                if not match:
                    tokens.append(s[i])
                    i+=1
            result.append(tokens)
        return result
    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        count=0 
        i=0 
        n=len(text)
        while i<n:
            match=False 
            for j in range(n,i,-1): 
                if text[i:j] in vocab:
                    count+=1 
                    i=j 
                    match=True 
                    break
            if not match:
                count+=1
                i+=1
        return count
        

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        words=text.split()
        if not words:
            return 0.0 

        total_tokens=self.count_tokens(text,vocab)
        return round(total_tokens/len(words),4)