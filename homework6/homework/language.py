from .models import LanguageModel, AdjacentLanguageModel, Bigram, load_model, TCN
from . import utils
import torch


def log_likelihood(model: LanguageModel, some_text: str):
    """
    Your code here

    Evaluate the log-likelihood of a given string.

    Hint: utils.one_hot might come in handy

    :param model: A LanguageModel
    :param some_text:
    :return: float
    """

    pred = model.predict_all(some_text)
    pred = pred[:, :len(some_text)]
    mask = utils.one_hot(some_text)
    result = torch.masked_select(pred, mask.bool())

    return torch.sum(result)
    # raise NotImplementedError('log_likelihood')


def avg_log_likelihood(model: LanguageModel, some_text: str):

    pred = model.predict_all(some_text)
    pred = pred[:, :len(some_text)]
    mask = utils.one_hot(some_text)
    result = torch.masked_select(pred, mask.bool())

    return torch.mean(result)


def sample_random(model: LanguageModel, max_length: int = 100):
    """
    Your code here.

    Sample a random sentence from the language model.
    Terminate once you reach a period '.'

    :param model: A LanguageModel
    :param max_length: The maximum sentence length
    :return: A string
    """
    s = str()
    length = 0
    stop = 0

    while length < max_length and not stop:

        pred = model.predict_all(s)
        if pred.size()[1] > 1:
            pred = pred[:, -1]

        m = torch.distributions.Categorical(logits=pred.squeeze())
        next = utils.vocab[m.sample()]
        s = s + next

        length += 1
        if next == '.':
            break

    return s
    # raise NotImplementedError('sample_random')


class TopNHeap:
    """
    A heap that keeps the top N elements around
    h = TopNHeap(2)
    h.add(1)
    h.add(2)
    h.add(3)
    h.add(0)
    print(h.elements)
    > [2,3]

    """
    def __init__(self, N):
        self.elements = []
        self.N = N

    def add(self, e):
        from heapq import heappush, heapreplace
        if len(self.elements) < self.N:
            heappush(self.elements, e)
        elif self.elements[0][0] < e[0]:
            heapreplace(self.elements, e)


def beam_search(model: LanguageModel, beam_size: int, n_results: int = 10, max_length: int = 100, average_log_likelihood: bool = False):
    """
    Your code here

    Use beam search for find the highest likelihood generations, such that:
      * No two returned sentences are the same
      * the `log_likelihood` of each returned sentence is as large as possible

    :param model: A LanguageModel
    :param beam_size: The size of the beam in beam search (number of sentences to keep around)
    :param n_results: The number of results to return
    :param max_length: The maximum sentence length
    :param average_log_likelihood: Pick the best beams according to the average log-likelihood, not the sum
                                   This option favors longer strings.
    :return: A list of strings of size n_results
    """
    from heapq import heappush, heapreplace, heappop

    heaps = []
    heaps.append(TopNHeap(beam_size))
    heaps[0].add((-100.0, ""))
    complete = TopNHeap(n_results)

    for k in range(1, max_length + 1):

        heaps.append(TopNHeap(beam_size))
        for j in heaps[k - 1].elements:

            s = j[1]
            if len(s) > 0 and s[-1] == '.':
                # heaps[k].add(j)
                continue

            for i in range(0, len(utils.vocab)):

                temp = s + utils.vocab[i]
                if average_log_likelihood:
                    score = avg_log_likelihood(model, temp)
                else:
                    score = log_likelihood(model, temp)

                tup = (score.item(), temp)
                if utils.vocab[i] == '.':
                    complete.add(tup)
                heaps[k].add(tup)

    res = []
    # for it in range(len(heaps[max_length].elements) - n_results):
    #     heappop(heaps[max_length].elements)
    #
    # for it in range(0, n_results):
    #     res.append(heappop(heaps[max_length].elements)[1])

    for it in range(0, n_results):
        res.append(heappop(complete.elements)[1])

    return res
    #raise NotImplementedError('beam_search')


if __name__ == "__main__":
    """
      Some test code.
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-m', '--model', choices=['Adjacent', 'Bigram', 'TCN'], default='Adjacent')
    args = parser.parse_args()

    lm = AdjacentLanguageModel() if args.model == 'Adjacent' else (load_model() if args.model == 'TCN' else Bigram())

    for s in ['abcdefg', 'abcgdef', 'abcbabc', '.abcdef', 'fedcba.']:
        print(s, float(log_likelihood(lm, s)))
    print()


    for i in range(10):
        s = sample_random(lm)
        print(s, float(log_likelihood(lm, s)) / len(s))
    print()

    for s in beam_search(lm, 100, max_length=4):
        # print(s, float(log_likelihood(lm, s)) / len(s))
        print(s, float(log_likelihood(lm, s)))
    print()

    tcn = TCN()
    i = 0
    input = torch.zeros(len(utils.vocab), 100)
    input[:, i] = float('NaN')

    one_hot = (torch.randint(len(utils.vocab), (1, 1, 10)) == torch.arange(len(utils.vocab))[None, :, None]).float()
    output = TCN(one_hot)
    # print(input[None, None].size())
    output = tcn(input[None])[0]
