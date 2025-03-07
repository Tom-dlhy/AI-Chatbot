import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 20000
eval_interval = 200
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(torch.cuda.get_device_name(0))

eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.5
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """
    Une tête unique de self-attention multi-têtes.

    Cette classe prend en entrée un tenseur de forme (batch_size, longueur_sequence, dimension_embedding) et effectue les opérations suivantes :
    1) Projette l’embedding de chaque token en vecteurs de clés (key), de requêtes (query) et de valeurs (value) de taille `head_size`.
    2) Applique un masque causal (triangulaire inférieur) pour que chaque token ne puisse s’appuyer que sur lui-même et les tokens précédents, préservant ainsi la propriété auto-régressive.
    3) Calcule les scores d’attention via le produit scalaire entre les requêtes et les clés, normalise ces scores avec la fonction softmax, puis applique une couche de dropout pour la régularisation.
    4) Utilise ces scores pour produire une somme pondérée des vecteurs de valeurs, fournissant ainsi une nouvelle représentation contextuelle pour chaque token.

    Pourquoi faire cela ?
    - Ce mécanisme permet à chaque token de se focaliser sélectivement sur les tokens les plus pertinents pour la prédiction suivante.
    - En utilisant plusieurs têtes d’attention, le modèle peut apprendre différents types de relations contextuelles en parallèle.

    Args:
        head_size (int): La dimension des projections key, query et value pour cette tête d’attention.

    Renvoie:
        Tensor: Une nouvelle représentation de chaque token, intégrant les informations contextuelles de tous les tokens auxquels il prête attention.
    """


    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """
    Attention multi-têtes en parallèle.

    Cette classe crée plusieurs têtes de self-attention indépendantes (instances de la classe Head)
    et combine leurs sorties pour obtenir une meilleure capacité de modélisation contextuelle.

    Fonctionnement :
    1) Chaque tête d’attention traite l’entrée indépendamment, apprenant différentes relations contextuelles.
    2) Les sorties de toutes les têtes sont concaténées le long de la dimension des canaux.
    3) Un projet linéaire (linear projection) et un dropout sont appliqués pour mélanger les informations des différentes têtes.

    Pourquoi plusieurs têtes ?
    - Avoir plusieurs têtes permet au modèle d’extraire en parallèle différents types de relations entre les tokens,
    par exemple, la relation de distance, de position, de type de mot, etc.

    Args:
        num_heads (int): Le nombre de têtes d’attention à utiliser.
        head_size (int): La dimension de chaque tête d’attention (taille des projections key, query, value).

    Renvoie:
        Tensor: La sortie fusionnée de toutes les têtes d’attention, de forme (batch_size, longueur_sequence, dimension_embedding).
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """
    Un réseau entièrement connecté pour compléter l'attention dans un bloc Transformer.

    Ce module est composé de deux couches linéaires séparées par une activation ReLU (Rectified Linear Unit),
    avec un facteur multiplicatif de 4, suivi d'un Dropout pour la régularisation.

    Fonctionnement :
    1) On projette d'abord l'entrée dans un espace 4 fois plus grand que n_embd.
    2) On applique la fonction d’activation ReLU pour introduire la non-linéarité.
    3) On re-projette dans l’espace d’origine (n_embd).
    4) Enfin, on applique du Dropout afin de limiter le surapprentissage.

    Pourquoi est-ce important ?
    - Cette couche combine les informations de l’attention et augmente la capacité de modélisation du réseau,
    en apprenant des relations non-linéaires complexes entre les tokens.

    Args:
        n_embd (int): La dimension de l’embedding ou le nombre de canaux d'entrée.

    Retourne:
        Tensor: Un tenseur de même taille que l’entrée, transformé par un réseau feed-forward.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Un bloc Transformer : il combine un mécanisme d'attention et un réseau feed-forward.

    Fonctionnement :
    1) On applique d’abord une normalisation par couche (LayerNorm) sur l’entrée, puis on calcule l’attention multi-têtes (MultiHeadAttention).
    2) On ajoute la sortie de l’attention aux embeddings d’origine (résidu) pour préserver l’information initiale.
    3) On applique à nouveau une normalisation par couche, suivie d’un passage dans le réseau feed-forward (FeedForward).
    4) On ajoute ensuite le résultat de ce feed-forward aux représentations existantes (autre connexion résiduelle).

    Pourquoi faire cela ?
    - Cette structure normalisée et résiduelle évite les problèmes de vanishing/exploding gradients et facilite
    l’apprentissage de relations complexes entre les tokens.

    Args:
        n_embd (int): La dimension d’embedding, soit le nombre de canaux dans les représentations.
        n_head (int): Le nombre de têtes d’attention (heads) souhaitées dans le mécanisme multi-têtes.

    Retourne:
        Tensor: La sortie transformée, de la même forme que l’entrée (batch_size, longueur_sequence, n_embd).
    """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
start_iter = 0


# ---------------- Execution Mode Selection ----------------
print("Choose a mode:")
print("1: Load the model for evaluation")
print("2: Retrain the model from scratch")
print("3: Continue training from a checkpoint")
mode = input("Your choice (1/2/3): ")


if mode == '1':
    # Load the model for evaluation only
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("Model loaded. Generating text:")
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
else:
    if mode == '3':
        # Continue training from the checkpoint
        checkpoint = torch.load('checkpoint.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_iter = checkpoint.get('iter', 0)
        print(f"Resuming training from iteration {start_iter}...")
    else:
        print("Training from scratch.")

# ---------------- Training Loop ----------------
    for iter in range(start_iter, max_iters):
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            # Save checkpoint
            torch.save({
                'iter': iter,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, 'checkpoint.pth')
        
        xb, yb = get_batch('train')
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

# ---------------- Final Text Generation ----------------
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("\nFinal generated text:")
print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
