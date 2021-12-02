from typing import Any
import torch as pt
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np
from cp_flatten import QuackConstants
import torch.nn.functional as F
import pytorch_lightning as pl


class AttentionScore(nn.Module):
    """
    Uses the dot-product to calculate the attention scores.
    See {Raff, Inside Deep Learning: Math, Algorithms, Models} Ch. 10
    """

    def __init__(self, dim: int) -> None:
        """

        Parameters
        ----------
        dim: int
            The dimension of the hidden state axis coming into the dot-product.
        """
        super().__init__()
        self.H = dim

    def forward(self, states: pt.Tensor, context: pt.Tensor) -> pt.Tensor:
        """
        Computes the dot-product score:

        :math`score(h_t, c) = \frac{h^T_t \cdot c}{\sqrt{H}}`

        Parameters
        ----------
        states: pt.Tensor
            Hidden states; shape (B, T, H)
        context: pt.Tensor
            Context values; shape (B, H)

        Returns
        -------
        pt.Tensor
            Scores for T items, based on context; shape (B, T, 1)
        """
        # T = states.size(1)
        scores = pt.bmm(states, context.unsqueeze(2)) / np.sqrt((self.H)) # (B, T, H) -> (B, T, 1)
        return scores


class AttentionModule(nn.Module):
    """
    Applies attention to the hidden states.
    Adapted from {Raff, Inside Deep Learning: Math, Algorithms, Models} Ch. 10
    """

    def __init__(self):
        super().__init__()

    def forward(self, states: pt.Tensor, attention_scores, mask=None):
        """
        states: pt.Tensor
            (B, T, H) shape giving the T different possible inputs
        attention_scores: (B, T, 1) score for each item at each context
        mask: None if all items are present. Else a boolean tensor of shape
            (B, T), with `True` indicating which items are present / valid.

        returns: a tuple with two tensors. The first tensor is the final context
        from applying the attention to the states (B, H) shape. The second tensor
        is the weights for each state with shape (B, T, 1).
        """

        if mask is not None:
            # set everything not present to a large negative value that will cause vanishing gradients
            attention_scores[~mask] = -1000.0
        # compute the weight for each score
        weights = F.softmax(attention_scores, dim=1)  # (B, T, 1) still, but sum(T) = 1

        final_context = (states * weights).sum(dim=1)  # (B, T, D) * (B, T, 1) -> (B, D)
        return final_context, weights


class QuackAutoEncoder(pl.LightningModule):
    """
    A Sequence-to-Sequence based autoencoder adapted from {Raff, Inside Deep Learning: Math, Algorithms, Models} Ch. 11.
    """
    def __init__(self, num_embeddings: int, embed_size: int, hidden_size: int, layers: int = 1,
                 max_decode_length: int = None, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.__hidden_size = hidden_size
        self.__max_decode_length = max_decode_length
        self.__learning_rate = 1e-3
        # Networks
        self.__embed = nn.Embedding(num_embeddings, embed_size, padding_idx=int(QuackConstants.XLMR_PAD.value))
        # We will use a bi-directional GRU so hidden size is divided by 2.
        self.__encoder = nn.GRU(input_size=embed_size, hidden_size=hidden_size // 2, num_layers=layers,
                                batch_first=True, bidirectional=True)
        # Decoder is uni-directional, and uses GRUCell.
        self.__decoder = nn.ModuleList(
            [nn.GRUCell(embed_size, hidden_size)]
            + [nn.GRUCell(hidden_size, hidden_size) for _ in range(layers - 1)]
        )
        # A small, fully connected, network that will combine attention and context to predict the next word value.
        self.__predict_word = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_embeddings)
        )
        # An attention network
        self.__attention = AttentionModule()
        self.__attention_score = AttentionScore(hidden_size)


    def mask_input(self, padded_input: pt.Tensor) -> pt.Tensor:
        """

        Parameters
        ----------
        padded_input: pt.Tensor
            The padded input (B, T)

        Returns
        -------
        pt.Tensor
            A boolean tensor (B, T).  True indicates the value at that time is usable, not fill.

        """
        with pt.no_grad():
            mask = padded_input.ne(QuackConstants.XLMR_PAD.value)
        return mask

    def loss_over_time(self, original: pt.Tensor, output: pt.Tensor) -> float:
        """
        Sum losses over time dimension, comparing original tokens and predicted tokens.

        Parameters
        ----------
        original: pt.Tensor
          The original input
        output: pt.Tensor
          The predicted output

        Returns
        -------
        float
          The aggregated loss.

        """
        cross_entropy = nn.CrossEntropyLoss(ignore_index=QuackConstants.XLMR_PAD.value)
        max_time = min(original.size(1), output.size(1))  # T
        loss = 0.0
        for time in range(max_time):
            loss += cross_entropy(output[:, time], original[:, time])
        return loss


    def forward(self, x: pt.Tensor) -> tuple[pt.Tensor, pt.Tensor]:
        """
        We put just the encoding process in forward.  The twin decoding process will only be found in
        the common step used in training and testing. This prepares the model for its intended use as
        as encoding and condensing latent features.

        Parameters
        ----------
        x: pt.Tensor
            The input, which should be (B, T) shaped.

        Returns
        -------
        tuple[pt.Tensor, pt.Tensor]
             * Final outputs of the encoding layer.
             * Encoded processing of the input tensor
        """
        # Store the dimensions
        batch_dim, temporal_dim = x.size()  # B, T
        mask = self.mask_input(x)
        # Calculate sequence lengths using the mask.
        lengths = mask.sum(dim=1).view(-1)  # Shape (B).
        # Transform to an Embed.
        x_embed = self.__embed(x) # Shape (B, T, D) which is what GRU expects.
        # GRU efficiency is increased with a PackedSequence object.
        x_packed = pack_padded_sequence(x_embed, lengths.cpu(), batch_first=True, enforce_sorted=False)
        encoded_sequence, last_h = self.__encoder(x_packed)
        encoded_sequence, _ = pad_packed_sequence(encoded_sequence)
        # encoded_sequence has shape (B, T, 2, D/2) since the GRU is bidirectional.
        encoded_sequence = encoded_sequence.view(batch_dim, temporal_dim, -1)  # encoded_sequence shape is now (B, T, D)
        hidden_size = encoded_sequence.size(2)
        last_h = last_h.view(-1, 2, batch_dim, hidden_size//2)[-1, :, :, :]  # last_h shape is now (2, B, D/2)
        last_h = last_h.permute(1, 0, 2).reshape(batch_dim, -1)  # last_h shape is now (B, D)
        return last_h, encoded_sequence

    def _common_step(self, x: pt.Tensor, batch_index: int, step_id: str) -> float:
        final_state, sequence = self.forward(x)

        # Now add the decoder part.
        decoder_depth = len(self.__decoder)
        # We will stack the predictions at the end.
        all_predictions = []
        # First, replicate the initial hidden state for each cell in the decoder.
        hidden_priors = [final_state for cell in range(decoder_depth)]
        # Prime the value of decoder input for the first iteration of the decoding loop.
        mask = self.mask_input(x)
        lengths = mask.sum(dim=1).view(-1)  # Shape (B).
        decoder_input = self.__embed(x.gather(1, lengths.view(-1, 1)-1).flatten()) #(B, D)
        # Calculate decoding steps:
        temporal_dim = x.size(1)  # T
        steps = min(self.__max_decode_length, temporal_dim)
        # Do we use teacher forcing (true) or auto-regressive (false)
        teacher_forcing = np.random.choice((True, False))
        for time in range(steps):
            x_in = decoder_input #(B, D)

            for layer in range(decoder_depth):
                hidden_prior = hidden_priors[layer]
                # Process through the decoder cells and store the result
                h = self.__decoder[layer](x_in, hidden_prior)
                hidden_priors[layer] = h
                x_in = h
            hidden_decoder = x_in  # (B, D), we now have the hidden state for the decoder at this time step.

            # Now we apply attention.
            scores = self.__attention_score(sequence, hidden_decoder)
            context, weights = self.__attention(sequence, scores, mask)
            # Now we use the prediction network to get the next predicted token in the decoding.
            prediction_input = pt.cat((context, hidden_decoder), dim=1)  # (B, D) + (B, D)  -> (B, 2*D)
            token_prediction = self.__predict_word(prediction_input)  # (B, 2*D) -> (B, V)
            all_predictions.append(token_prediction)
            # Now select the input of the next time step.  No gradient on the input tokens.
            with pt.no_grad():
                if teacher_forcing:
                    next_words = x[:, time].squeeze()
                else:
                    # Sample the next token based on the predictions made
                    next_words = pt.multinomial(F.softmax(token_prediction, dim=1), 1)[:, -1]
            decoder_input = self.__embed(next_words)
        predicted_batch = pt.stack(all_predictions, dim=1)
        loss = self.loss_over_time(x, predicted_batch)
        self.log(f"{step_id}_loss", loss)
        return loss

    def training_step(self, x: pt.Tensor, batch_index: int) -> dict:
        return {'loss': self._common_step(x, batch_index, 'train')}

    def validation_step(self, x: pt.Tensor, batch_index: int) -> float:
        return self._common_step(x, batch_index, 'val')

    def test_step(self, x: pt.Tensor, batch_index: int) -> float:
        return self._common_step(x, batch_index, 'test')

    def configure_optimizers(self) -> pt.optim.Optimizer:
        return pt.optim.AdamW(self.parameters(), lr=self.__learning_rate)
