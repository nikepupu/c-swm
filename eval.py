import argparse
import torch
import utils
import os
import pickle


from torch.utils import data
import numpy as np
from collections import defaultdict

import modules
from torch.nn.utils.rnn import pad_sequence
torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    pre_image, action, post_image, low_descs, task_desc = zip(*batch)

    low_descs = [torch.LongTensor(i) for i in low_descs]
    task_desc = [torch.LongTensor(i) for i in task_desc]

    low_descs = pad_sequence(low_descs, batch_first=True, padding_value=1)
    task_desc = pad_sequence(task_desc, batch_first=True, padding_value=1)
    return torch.stack(pre_image, 0), torch.tensor(action), torch.stack(post_image, 0), low_descs, task_desc

parser = argparse.ArgumentParser()
parser.add_argument('--save-folder', type=str,
                    default='checkpoints',
                    help='Path to checkpoints.')
parser.add_argument('--num-steps', type=int, default=1,
                    help='Number of prediction steps to evaluate.')
parser.add_argument('--dataset', type=str,
                    default='data/shapes_eval.h5',
                    help='Dataset string.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA training.')

args_eval = parser.parse_args()


meta_file = os.path.join(args_eval.save_folder, 'metadata.pkl')
model_file = os.path.join(args_eval.save_folder, 'model.pt')

args = pickle.load(open(meta_file, 'rb'))['args']
print(args)
args.cuda = not args_eval.no_cuda and torch.cuda.is_available()
args.batch_size = 64
args.dataset = args_eval.dataset
args.seed = 0

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

device = torch.device('cuda' if args.cuda else 'cpu')

# dataset = utils.PathDataset(
#     hdf5_file=args.dataset, path_length=args_eval.num_steps)
dataset = utils.ThorTransitionsDataset('/home/steven/dataset_validation_action')
eval_loader = data.DataLoader(
    dataset, batch_size=args.batch_size, collate_fn = collate_fn, shuffle=False, num_workers=4)

# Get data sample
obs = eval_loader.__iter__().next()[0]
input_shape = obs[0].size()
print(input_shape)
model = modules.ContrastiveSWM(
    embedding_dim=args.embedding_dim,
    hidden_dim=args.hidden_dim,
    action_dim=args.action_dim,
    input_dims=input_shape,
    num_objects=args.num_objects,
    sigma=args.sigma,
    hinge=args.hinge,
    ignore_action=args.ignore_action,
    copy_action=args.copy_action,
    encoder=args.encoder).to(device)
print(model_file)
model.load_state_dict(torch.load(model_file))
model.eval()

topk = [1, 5, 10]
# topk = [1]
hits_at = defaultdict(int)
num_samples = 0
rr_sum = 0

pred_states = []
next_states = []

with torch.no_grad():

    for batch_idx, data_batch in enumerate(eval_loader):
        # data_batch = [[t.to(
        #     device) for t in tensor] for tensor in data_batch]
        # print(data_batch)
        obs, actions, next_obs, low_descs, task_desc = data_batch
        obs = obs.cuda()
        actions = actions.cuda()
        next_obs = next_obs.cuda()
        low_descs = low_descs.cuda()
        task_desc = task_desc.cuda()

        language_encoding1, _ = model.lstm(model.embed(task_desc) )
        language_encoding2, _ = model.lstm(model.embed(low_descs))
        language_encoding = language_encoding1[:,-1,:] + language_encoding2[:,-1,:]
       # if observations[0].size(0) != args.batch_size:
        #     continue

        # obs = observations[0]
        # next_obs = observations[-1]

        state = model.obj_encoder(model.obj_extractor(obs), language_encoding)
        next_state = model.obj_encoder(model.obj_extractor(next_obs), language_encoding)
        
        pred_state = state
        for i in range(args_eval.num_steps):
            pred_trans = model.transition_model(pred_state, actions)
            pred_state = pred_state + pred_trans

        pred_states.append(pred_state.cpu())
        next_states.append(next_state.cpu())

    pred_state_cat = torch.cat(pred_states, dim=0)
    next_state_cat = torch.cat(next_states, dim=0)

    full_size = pred_state_cat.size(0)

    # Flatten object/feature dimensions
    next_state_flat = next_state_cat.view(full_size, -1)
    pred_state_flat = pred_state_cat.view(full_size, -1)

    dist_matrix = utils.pairwise_distance_matrix(
        next_state_flat, pred_state_flat)
    dist_matrix_diag = torch.diag(dist_matrix).unsqueeze(-1)
    dist_matrix_augmented = torch.cat(
        [dist_matrix_diag, dist_matrix], dim=1)

    # Workaround to get a stable sort in numpy.
    dist_np = dist_matrix_augmented.numpy()
    indices = []
    for row in dist_np:
        keys = (np.arange(len(row)), row)
        indices.append(np.lexsort(keys))
    indices = np.stack(indices, axis=0)
    indices = torch.from_numpy(indices).long()

    print('Processed {} batches of size {}'.format(
        batch_idx + 1, args.batch_size))

    labels = torch.zeros(
        indices.size(0), device=indices.device,
        dtype=torch.int64).unsqueeze(-1)

    num_samples += full_size
    print('Size of current topk evaluation batch: {}'.format(
        full_size))

    for k in topk:
        match = indices[:, :k] == labels
        num_matches = match.sum()
        hits_at[k] += num_matches.item()

    match = indices == labels
    _, ranks = match.max(1)

    reciprocal_ranks = torch.reciprocal(ranks.double() + 1)
    rr_sum += reciprocal_ranks.sum()

    pred_states = []
    next_states = []

for k in topk:
    print('Hits @ {}: {}'.format(k, hits_at[k] / float(num_samples)))

print('MRR: {}'.format(rr_sum / float(num_samples)))
