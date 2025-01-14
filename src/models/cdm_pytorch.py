import numpy as np
import sys
import torch as t
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, '../utils/')
#import choice_utils as cu
import time
import pdb
from tqdm import tqdm

class Embedding(nn.Module):
    """
    Redefining torch.nn.Embedding (see docs for that function)
    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, _weight=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx

        if _weight is None:
            self.weight = nn.Parameter(t.randn([self.num_embeddings, self.embedding_dim])/np.sqrt(self.num_embeddings))
        else:
            assert list(_weight.shape) == [num_embeddings, embedding_dim], \
                'Shape of weight does not match num_embeddings and embedding_dim'
            self.weight = nn.Parameter(_weight)

        if self.padding_idx is not None:
            with t.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def forward(self, x):
        if self.padding_idx is not None:
            with t.no_grad():
                self.weight[self.padding_idx].fill_(0)

        return self.weight[x]

class DataLoader():
    """
    Redefining torch.utils.data.DataLoader, see docs for that function
    Done so because it is faster for CPU only use.
    """
    def __init__(self, data, batch_size=None, shuffle=False):
        # data must be a list of tensors
        self.data = data
        self.data_size = data[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = 0
        self.stop_iteration_flag = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_iteration_flag:
            self.stop_iteration_flag = False
            raise StopIteration()
        if self.batch_size is None or self.batch_size == self.data_size:
            self.stop_iteration_flag = True
            return self.data
        else:
            i = self.counter
            bs = self.batch_size
            self.counter += 1
            batch = [item[i * bs:(i + 1) * bs] for item in self.data]
            if self.counter * bs >= self.data_size:
                self.counter = 0
                self.stop_iteration_flag = True
                if self.shuffle:
                    random_idx = np.arange(self.data_size)
                    np.random.shuffle(random_idx)
                    self.data = [item[random_idx] for item in self.data]
            return batch

class ChoiceModel(nn.Module):
    """
    An abstract class for an arbitrary choice model
    """
    def __init__(self, num_items, ism):
        """
        Initializes the Choice Model
        Inputs: 
        num_items - the number of items in the choice system modeled
        ism - if dataset is multi-set, in which case padding is used
        """
        super().__init__()
        self.num_items = num_items
        self.ism = ism

    def loss_func(self, y_hat, y, x_lengths=None):
        """
        Evaluates the Choice Model
        Inputs: 
        y_hat - the log softmax values that come from the forward function
        y - actual labels - the choice in a set (must be less than x_lengths)
        x_lengths - the size of the choice set, used to determine padding. 
        The current implementation assumes that y are less than x_lengths, so this
        is unused.
        """
        return F.nll_loss(y_hat, y[:, None].long())

    def acc_func(self, y_hat, y, x_lengths=None):
        return (y_hat.argmax(1).int() == y[:,None].int()).float().mean()

class MNL(ChoiceModel):
    """
    The MNL model
    """
    def __init__(self, num_items, ism):
        """
        Initializes the MNL
        Inputs: 
        num_items - the number of items in the choice system modeled
        ism - if dataset is multi-set, in which case padding is used
        """
        super().__init__(num_items, ism)

        padding_idx = self.num_items
        self.logits = Embedding(
            num_embeddings=self.num_items + 1,  # +1 for the padding
            embedding_dim=1,
            padding_idx=padding_idx
        )

    def forward(self, x, x_lengths=None, inf_weight=float('-inf')):
        """
        Computes log probabilities using the MNL
        Inputs: 
        x - item indices involved in the CDM set of interest. size: batch_size x
        maximum sequence length
        x_lengths - size sizes of input. Used to determine padding
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len = x.size()
        logits = self.logits(x)

        if self.ism:
            logits[t.arange(seq_len)[None, :] >= x_lengths[:, None]] = inf_weight
        return F.log_softmax(logits, 1)

class NoiseMNL(ChoiceModel):
    """
    A Noisy MNL model
    """
    def __init__(self, num_items, epsilon, ism):
        """
        Initializes the MNL
        Inputs: 
        num_items - the number of items in the choice system modeled
        epsilon - parameter of uniform mixture
        ism - if dataset is multi-set, in which case padding is used

        """
        super().__init__(num_items, ism)

        padding_idx = self.num_items
        self.logits = Embedding(
            num_embeddings=self.num_items + 1,  # +1 for the padding
            embedding_dim=1,
            padding_idx=padding_idx
        )

    def forward(self, x, x_lengths=None, inf_weight=float('-inf')):
        """
        Computes log probabilities using the MNL
        Inputs: 
        x - item indices involved in the CDM set of interest. size: batch_size x
        maximum sequence length
        x_lengths - size sizes of input. Used to determine padding
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len = x.size()
        logits = self.logits(x)
        perturbs = t.ones_like(logits, dtype=float)/x_lengths[:,None]

        if self.ism:
            logits[t.arange(seq_len)[None, :] >= x_lengths[:, None]] = inf_weight
            perturbs[t.arange(seq_len)[None, :] >= x_lengths[:, None]] = 0
        return t.log(F.softmax(logits, 1)*(1-epsilon) + perturbs*epsilon)

class CDM(ChoiceModel):
    """
    The CDM model, described in "Discovering Context Effects from Raw Choice Data"
    """
    def __init__(self, num_items, ism, embedding_dim, no_ego=True):
        """
        Initializes the CDM
        Inputs: 
        num_items - the number of items in the choice system modeled (n in the paper)
        embedding_dim - dimension of CDM (r in the paper)
        ism - if dataset is multi-set, in which case padding is used
        no_ego - does not include the "self contextual contribution" t_x^Tc_x. This
                is always set to True as far as the paper is concerned.
        """
        super().__init__(num_items, ism)
        self.embedding_dim = embedding_dim
        self.no_ego = no_ego
        self.__build_model()

    def __build_model(self):
        """
        Helper function to initialize the CDM
        """    
        padding_idx = self.num_items  # CHANGE to be something more general?
        # Fix weight init
        self.target_embedding = Embedding(
            num_embeddings=self.num_items+1,  # +1 for the padding
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
            _weight=t.zeros([self.num_items+1, self.embedding_dim])
        )
        self.context_embedding = Embedding(
            num_embeddings=self.num_items + 1,  # +1 for the padding
            embedding_dim=self.embedding_dim,
            padding_idx=padding_idx,
            #_weight=t.zeros([self.num_items+1, self.embedding_dim])
        )
        # self.layer1 = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, x, x_lengths=None, inf_weight=float('-inf')):
        """
        Computes using the CDM
        Inputs: 
        x - item indices involved in the CDM set of interest. size: batch_size x
        maximum sequence length
        x_lengths - size sizes of input. Used to determine padding
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len = x.size()
        context_vecs = self.context_embedding(x) #self.layer1(self.target_embedding(x))

        if self.no_ego:
            context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
            logits = (self.target_embedding(x) * context_vecs).sum(-1,keepdim=True)
        else:
            context_vecs = context_vecs.sum(-2)[...,None]
            logits = self.target_embedding(x) @ context_vecs

        if self.ism:
            logits[t.arange(seq_len)[None, :] >= x_lengths[:, None]] = inf_weight
        return F.log_softmax(logits, 1)


class FullCDM(ChoiceModel):
    """
    The Full CDM model, described in "Discovering Context Effects from Raw Choice Data"
    """
    def __init__(self, num_items, ism, no_ego=True):
        """
        Initializes the Full CDM
        Inputs: 
        num_items - the number of items in the choice system modeled (n in the paper)
        ism - if dataset is multi-set, in which case padding is used
        no_ego - does not include the "self contextual contribution" t_x^Tc_x. This
                is always set to True as far as the paper is concerned.
        """
        super().__init__(num_items, ism)
        self.no_ego = no_ego
        padding_idx = self.num_items
        self.U = Embedding(
            num_embeddings=self.num_items + 1,  # +1 for the padding
            embedding_dim=self.num_items + 1, # +1 for the padding
            padding_idx=padding_idx
        )

    def forward(self, x, x_lengths=None, inf_weight=float('-inf')):
        """
        Computes using the Full CDM
        Inputs: 
        x - item indices involved in the CDM set of interest. size: batch_size x
        maximum sequence length
        x_lengths - size sizes of input. Used to determine padding
        inf_weight - used to "zero out" padding terms. Should not be changed.
        """    
        batch_size, seq_len = x.size()
        context_vecs = self.U(x) #self.layer1(self.target_embedding(x))
        x_0 = t.arange(batch_size)[:,None].expand([batch_size, seq_len])
        if self.no_ego:
            context_vecs = context_vecs.sum(-2, keepdim=True) - context_vecs
            x_1 = t.arange(seq_len)[None,:].expand([batch_size, seq_len])
            logits = context_vecs[x_0,x_1, x]
        else:
            context_vecs = context_vecs.sum(-2)[...,None]
            logits = context_vecs[x_0, x]
        logits = logits[...,None]
        if self.ism:
            logits[t.arange(seq_len)[None, :] >= x_lengths[:, None]] = inf_weight
        return F.log_softmax(logits, 1)


def get_model(Model, num_items, ism, lr, wd=0, seed=None, **kwargs):
    if seed is not None:
        t.manual_seed(seed)
    model = Model(num_items, ism, **kwargs)
    return model, t.optim.Adam(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)

def random_split(dataset, split_sizes, seed=None):
    # A rewrite of t.utils.data.random_split
    # Input:
        # dataset - iterable of tensors of equal size
        # split_sizes - iterable of integers, need to sum to dataset size
    # Output:
        # split_datasets - same length as split_sizes, with each entry having the number

    splits = np.cumsum(list(split_sizes))
    assert np.all(np.array([a.shape[0] for a in dataset])==splits[-1])
    random_idx = np.arange(splits[-1])
    if seed is not None:
        current_random_state = np.random.get_state()
        np.random.seed(seed)  # set seed according to specification
        np.random.shuffle(random_idx)
        np.random.set_state(current_random_state)  # restore to where it was pre-seed
    else: # No seed, so choose randomly
        np.random.shuffle(random_idx)

    split_idxs = np.split(random_idx, splits)[:-1]

    return [[a[split_idx] for a in dataset] for split_idx in split_idxs]


def load_data(dataset=None, dd=None, dm=None, ism=True, extra_name='choice_set_lengths',
              target_name='slot_chosen', seed=None):
    if dataset is not None:
        dd, dm, _, ism = cu.read_and_id_data(dataset)
    data_size, num_items = len(dd[target_name]), len(dm)

    whole_ds = list(map(t.tensor, [dd['context_ids'], dd[extra_name], dd[target_name]]))
    train_ds, val_ds = random_split(whole_ds, [int(.8*data_size), data_size-int(.8*data_size)], seed=seed)

    return train_ds, val_ds, dm, num_items, ism

def get_data(train_ds, val_ds, batch_size=None):
    # Note: can change val_bs to 2* batch_size if ever becomes a problem
    if batch_size is not None:
        tr_bs, val_bs = (batch_size, len(val_ds[0]))
    else: 
        tr_bs, val_bs = (len(train_ds[0]), len(val_ds[0]))

    train_dl = DataLoader(train_ds, batch_size=tr_bs, shuffle=batch_size is not None)
    val_dl = DataLoader(val_ds, batch_size=val_bs)
    return train_dl, val_dl

def ds_to_OJ(ds, n):
    context_ids, choice_set_lens, slot_chosen = ds
    m = len(context_ids)
    O = np.zeros([m, n+1])
    J = np.zeros([m, n])
    j_idx = context_ids[t.arange(m), slot_chosen].numpy()
    o_idx = ds[0].numpy().flatten()
    J[np.arange(m), j_idx] = 1
    O[np.tile(np.arange(m)[:,None],3).flatten(), o_idx] = 1
    O = O[:,:-1]
    
    assert np.all(O.sum(-1) == choice_set_lens.numpy()), 'something went wrong with O'
    assert np.all(J.sum(-1) == 1), 'something went wrong with J'
    
    return O,J

def OJ_to_ds(O, J):
    m,n = O.shape
    choice_set_lens = np.int64(O.sum(-1))
    k_max = choice_set_lens.max()
    context_ids = np.full([m, k_max], fill_value=n, dtype=np.long)
    
    o_nnz = O.nonzero()
    slots = np.zeros([m, k_max], dtype=np.long)
    idx = np.arange(k_max)[None, :] < choice_set_lens[:, None]

    slots[idx] = J[o_nnz[0],o_nnz[1]]
    slot_chosen = t.from_numpy(slots.nonzero()[1])

    context_ids[idx] = o_nnz[1]

    choice_set_lens = t.from_numpy(choice_set_lens)
    context_ids = t.from_numpy(context_ids)

    return [context_ids, choice_set_lens, slot_chosen]


def loss_batch(model, xb, yb, xlb, opt=None, retain_graph=None):
    if opt is not None:
        loss = model.loss_func(model(xb, xlb), yb, xlb)

        loss.backward(retain_graph=retain_graph)
        opt.step()
        opt.zero_grad()
    else:
        with t.no_grad():
            loss = model.loss_func(model(xb, xlb), yb, xlb)

    return loss

def acc_batch(model, xb, yb, xlb):
    with t.no_grad():
        return model.acc_func(model(xb, xlb), yb, xlb)

def fit(epochs, model, opt, train_dl, val_dl=None, verbose=False):
    val_loss = t.zeros(1)
    verbosify = tqdm if verbose else lambda x: x 
    for epoch in verbosify(range(epochs)):
        model.train()  # good practice because these are used by nn.BatchNorm2d and nn.Dropout
        counter = 0
        for xb, xlb, yb in train_dl:
            loss = loss_batch(model, xb, yb, xlb, opt, retain_graph=None if epoch != epochs - 1 else True)
            counter += 1
            if counter % 1000 == 0:
                print(f"counter hit, step: {counter}, loss: {loss}")
        if val_dl is not None:
            model.eval() # good practice like model.train()
            val_loss = [loss_batch(model, xb, yb, xlb) for xb, xlb, yb in val_dl]
            val_loss = sum(val_loss)/len(val_loss)
            # val_acc = [acc_batch(model, xb, yb, xlb) for xb, xlb, yb in val_dl]
            # val_acc = sum(val_acc) / len(val_acc)

        # print(f'Epoch: {epoch}, Training Loss: {loss}, Val Loss: {val_loss}, \
                # Val Accuracy {val_acc}')

    loss.backward() # for last gradient value
    with t.no_grad():
        gv = t.stack([(item.grad**2).sum() for item in model.parameters()]).sum()
    return loss.detach().numpy(), val_loss.numpy(), gv.numpy()

def eval(O,J, model, batch_size=None):
    val_ds = OJ_to_ds(O,J)
    val_bs = batch_size if batch_size is not None else len(val_ds[0])
    
    val_dl = DataLoader(val_ds, batch_size=val_bs)

    model.eval()
    val_loss = [loss_batch(model, xb, yb, xlb) for xb, xlb, yb in val_dl]

    return val_loss

def l2err_run(O,J, ism=True, batch_size=None, epochs=500, lr=5e-3, seed=2, wd=0, Model=CDM, verbose=False, **kwargs):
    # embedding_dim=5
    ds = OJ_to_ds(O,J)
    tr_bs = batch_size if batch_size is not None else len(ds[0])
    dl = DataLoader(ds, batch_size=tr_bs, shuffle=batch_size is not None)
    model, opt = get_model(Model, num_items=O.shape[1], ism=ism, lr=lr, wd=wd, 
        seed=seed, **kwargs)
    s = time.time()
    tr_loss, _, gv = fit(epochs, model, opt, dl, verbose=verbose)
    print(f'Runtime: {time.time() - s}')

    return model, tr_loss, gv


def compute_predicted_rankings_CDM(X,Y,model, epochs=20, batch_size=1, verbose=True, print_batches=1000, opt='Adam'):
    """
    Fits a choice model with pytorch's SGD

    X- Indicator vectors for choice sets
    Y- indices of choices
    model- tuple of (name of model, choice model to fit, parameters of the model)
    criterion- which loss function to use (default to negative log likelihood for MLE)
    epochs- number of times to loop over the training data
    batch_size- how large to make batches
    verbose- whether to print updates as training goes on
    print_batches- how often to print updates on training
    opt- which optimizer to use 'SGD' or 'Adam'
    """

    Y_X = np.zeros_like(X)
    Y_X[np.arange(len(Y)), Y] = 1
    model, tr_loss, gv = l2err_run(X, Y_X, batch_size=batch_size, epochs=epochs, 
        Model=model[1], seed=2, lr=5e-3, verbose=True, **model[2])

    return model