import numpy as np
import torch
import torch.optim as optim
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import random
from datetime import datetime
from collections import defaultdict

from model import HMGN_GNN
from data_loader import DataLoader



def hit_rate_at_k(user2scores, user2labels, K=1):
    hits, cnt = 0, 0
    for u, scores in user2scores.items():
        labels = user2labels[u]
        if labels.sum() == 0:
            continue
        idx = np.argsort(-scores)[:K]
        topk_labels = labels[idx]
        hits += int(topk_labels.sum() > 0)
        cnt += 1
    return hits / cnt if cnt > 0 else 0.0


def build_user_pos(df):

    upos = defaultdict(set)
    for _, r in df.iterrows():
        if int(r['label']) == 1:
            upos[int(r['userID'])].add(int(r['itemID']))
    return upos


def build_eval_candidates(x_test, all_items, user_pos_all, n_neg=49):

    user_pos_test = build_user_pos(x_test)
    user2cands, user2labels = {}, {}
    all_items_set = set(all_items)

    for u, pos_items_test in user_pos_test.items():
        if len(pos_items_test) == 0:
            continue

        pos_item = random.choice(list(pos_items_test))


        seen = user_pos_all.get(u, set())
        neg_pool = list(all_items_set - seen)
        if len(neg_pool) == 0:
            neg_pool = list(all_items_set)


        n_sample = min(n_neg, len(neg_pool))
        neg_items = random.sample(neg_pool, k=n_sample)


        cands = [pos_item] + neg_items


        labels = np.zeros(len(cands), dtype=np.int64)
        labels[0] = 1

        user2cands[u] = np.array(cands, dtype=np.int64)
        user2labels[u] = labels

    return user2cands, user2labels


@torch.no_grad()
def score_candidates(model, device, user2cands, batch_size=1024):

    model.eval()
    user2scores = {}
    for u, items in user2cands.items():

        users = np.full_like(items, fill_value=u)

        scores_collected = []
        start = 0
        while start < len(items):
            end = min(start + batch_size, len(items))
            u_batch = torch.as_tensor(users[start:end], device=device, dtype=torch.long)
            i_batch = torch.as_tensor(items[start:end], device=device, dtype=torch.long)
            s = model(u_batch, i_batch)
            s = s.view(-1).detach().cpu().numpy()
            scores_collected.append(s)
            start = end

        user2scores[u] = np.concatenate(scores_collected, axis=0)
    return user2scores




def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_loader = DataLoader(args.dataset)
    kg = data_loader.load_kg()
    df_dataset = data_loader.load_dataset()


    def train_test_val_split(data, ratio_train, ratio_test, ratio_val):
        train, middle = train_test_split(
            data,
            train_size=ratio_train,
            test_size=ratio_test + ratio_val,
            random_state=2023
        )
        ratio = ratio_val / (1 - ratio_train)
        test, validation = train_test_split(
            middle,
            test_size=ratio,
            random_state=2023
        )
        return (
            train.reset_index(drop=True),
            test.reset_index(drop=True),
            validation.reset_index(drop=True),
        )


    base_train, x_test, x_validation = train_test_val_split(df_dataset, 0.6, 0.2, 0.2)


    train_ratio = getattr(args, "train_ratio", 1.0)
    if not (0 < train_ratio <= 1.0):
        raise ValueError(f"train_ratio 必须在 (0,1] 内，当前为 {train_ratio}")

    if train_ratio < 1.0:
        x_train = base_train.sample(
            frac=train_ratio,
            random_state=2023
        ).reset_index(drop=True)
    else:
        x_train = base_train

    print(f"base_train: {len(base_train)}, "
          f"used_train (ratio={train_ratio}): {len(x_train)}, "
          f"val: {len(x_validation)}, test: {len(x_test)}")


    UAndI = dict()
    df_len = len(x_train['userID'])
    for _, row in tqdm(x_train.iterrows(), total=df_len, desc='Obtain items'):
        userID = row['userID']
        itemID = row['itemID']
        label = row['label']
        if label == 1:
            if userID in UAndI:
                UAndI[userID].append(itemID)
            else:
                UAndI[userID] = [itemID]

    UAndI_len = len(UAndI)
    for k, v in tqdm(UAndI.items(), total=UAndI_len, desc='Organize items'):
        v_len = len(v)
        if v_len < args.neighbor_sample_size:
            v.extend(random.choices(v, k=args.neighbor_sample_size - v_len))
            UAndI[k] = v
        elif v_len > args.neighbor_sample_size:
            UAndI[k] = random.sample(v, args.neighbor_sample_size)

    class PIFSA_GNN_Dataset(torch.utils.data.Dataset):
        def __init__(self, df):
            self.df = df

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            user_id = np.array(self.df.iloc[idx]['userID'])
            item_id = np.array(self.df.iloc[idx]['itemID'])
            label = np.array(self.df.iloc[idx]['label'], dtype=np.float32)
            return user_id, item_id, label

    train_dataset = PIFSA_GNN_Dataset(x_train)
    validation_dataset = PIFSA_GNN_Dataset(x_validation)
    test_dataset = PIFSA_GNN_Dataset(x_test)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=args.batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size
    )

    num_user, num_entity, num_relation = data_loader.get_num()
    net = HMGN_GNN(num_user, num_entity, num_relation, kg, args, device, UAndI).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.l2_weight)
    print('device:', device)

    auc_score_list, f1_score_list, precision_score_list, recall_score_list = [], [], [], []
    auc_score_t_list, f1_score_t_list, precision_score_t_list, recall_t_score_list = [], [], [], []

    train_loader_len = len(train_loader)
    validation_loader_len = len(validation_loader)
    test_loader_len = len(test_loader)

    hit_hist = []


    user_pos_all = build_user_pos(df_dataset[['userID', 'itemID', 'label']])
    all_items = df_dataset['itemID'].unique().tolist()
    n_neg_eval = getattr(args, "n_neg_eval", 49)
    users_per_eval = getattr(args, "users_per_eval", 1000)


    user2cands_full, user2labels_full = build_eval_candidates(
        x_test[['userID', 'itemID', 'label']],
        all_items,
        user_pos_all,
        n_neg=n_neg_eval
    )
    all_users_full = list(user2cands_full.keys())
    if len(all_users_full) == 0:
        raise ValueError("测试集中没有带正样本的用户，无法计算 Hit@K。")

    for epoch in range(args.n_epochs):
        net.train()
        for i, (user_ids, item_ids, labels) in tqdm(
            enumerate(train_loader),
            total=train_loader_len,
            desc='Train'
        ):
            user_ids, item_ids, labels = (
                user_ids.to(device),
                item_ids.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            outputs = net(user_ids, item_ids).to(device)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            total_roc = total_f1 = total_precision = total_recall = 0.0
            net.eval()
            for user_ids, item_ids, labels in tqdm(
                validation_loader,
                total=validation_loader_len,
                desc='Validation'
            ):
                user_ids, item_ids, labels = (
                    user_ids.to(device),
                    item_ids.to(device),
                    labels.to(device),
                )
                outputs = net(user_ids, item_ids).to(device)
                pre_label = (outputs >= 0.5).long().cpu().numpy()
                try:
                    total_roc += roc_auc_score(
                        labels.cpu().numpy(),
                        outputs.cpu().numpy()
                    )
                    total_f1 += f1_score(
                        labels.cpu().numpy(),
                        pre_label,
                        average='binary'
                    )
                    total_precision += precision_score(
                        labels.cpu().numpy(),
                        pre_label,
                        average='binary'
                    )
                    total_recall += recall_score(
                        labels.cpu().numpy(),
                        pre_label,
                        average='binary'
                    )
                except ValueError:
                    pass
            auc_score_list.append(total_roc / max(1, len(validation_loader)))
            f1_score_list.append(total_f1 / max(1, len(validation_loader)))
            precision_score_list.append(total_precision / max(1, len(validation_loader)))
            recall_score_list.append(total_recall / max(1, len(validation_loader)))

        # Test (batch metrics)
        with torch.no_grad():
            total_t_roc = total_t_f1 = total_t_precision = total_t_recall = 0.0
            net.eval()
            for user_ids, item_ids, labels in tqdm(
                test_loader,
                total=test_loader_len,
                desc='Test'
            ):
                user_ids, item_ids, labels = (
                    user_ids.to(device),
                    item_ids.to(device),
                    labels.to(device),
                )
                outputs = net(user_ids, item_ids).to(device)
                pre_label = (outputs >= 0.5).long().cpu().numpy()
                try:
                    total_t_roc += roc_auc_score(
                        labels.cpu().numpy(),
                        outputs.cpu().numpy()
                    )
                    total_t_f1 += f1_score(
                        labels.cpu().numpy(),
                        pre_label,
                        average='binary'
                    )
                    total_t_precision += precision_score(
                        labels.cpu().numpy(),
                        pre_label,
                        average='binary'
                    )
                    total_t_recall += recall_score(
                        labels.cpu().numpy(),
                        pre_label,
                        average='binary'
                    )
                except ValueError:
                    pass
            auc_score_t_list.append(total_t_roc / max(1, len(test_loader)))
            f1_score_t_list.append(total_t_f1 / max(1, len(test_loader)))
            precision_score_t_list.append(total_t_precision / max(1, len(test_loader)))
            recall_t_score_list.append(total_t_recall / max(1, len(test_loader)))


        with torch.no_grad():
            net.eval()
            rng = np.random.RandomState(2025 + epoch)

            # 抽样用户
            if len(all_users_full) > users_per_eval:
                user_sample = rng.choice(
                    all_users_full, size=users_per_eval, replace=False
                ).tolist()
            else:
                user_sample = all_users_full

            user2cands = {u: user2cands_full[u] for u in user_sample}
            user2labels = {u: user2labels_full[u] for u in user_sample}

            user2scores = score_candidates(net, device, user2cands, batch_size=4096)

            hit1 = hit_rate_at_k(user2scores, user2labels, K=1)
            hit5 = hit_rate_at_k(user2scores, user2labels, K=5)
            hit10 = hit_rate_at_k(user2scores, user2labels, K=10)

            hit_res_epoch = {
                "Hit@1": hit1,
                "Hit@5": hit5,
                "Hit@10": hit10,
                "users_evaluated": len(user_sample),
            }
            hit_hist.append(hit_res_epoch)

            print(
                f"   [Epoch {epoch + 1}] "
                f"Hit@1: {hit1:.4f}  "
                f"Hit@5: {hit5:.4f}  "
                f"Hit@10: {hit10:.4f}  "
                f"Users: {len(user_sample)}"
            )

        print(
            '\n' + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            f'[Epoch {epoch + 1}] '
            f'Val auc:{auc_score_list[-1]:.4f} '
            f'f1:{f1_score_list[-1]:.4f} '
            f'pre:{precision_score_list[-1]:.4f} '
            f'recall:{recall_score_list[-1]:.4f} '
            f'Test auc:{auc_score_t_list[-1]:.4f} '
            f'f1:{f1_score_t_list[-1]:.4f} '
            f'pre:{precision_score_t_list[-1]:.4f} '
            f'recall:{recall_t_score_list[-1]:.4f}'
        )

    with torch.no_grad():
        net.eval()
        rng = np.random.RandomState(2025)
        if len(all_users_full) > users_per_eval:
            user_sample_final = rng.choice(
                all_users_full, size=users_per_eval, replace=False
            ).tolist()
        else:
            user_sample_final = all_users_full

        user2cands_final = {u: user2cands_full[u] for u in user_sample_final}
        user2labels_final = {u: user2labels_full[u] for u in user_sample_final}
        user2scores_final = score_candidates(net, device, user2cands_final, batch_size=4096)

        hit1 = hit_rate_at_k(user2scores_final, user2labels_final, K=1)
        hit5 = hit_rate_at_k(user2scores_final, user2labels_final, K=5)
        hit10 = hit_rate_at_k(user2scores_final, user2labels_final, K=10)

        hit_res = {
            "Hit@1": hit1,
            "Hit@5": hit5,
            "Hit@10": hit10,
            "users_evaluated": len(user_sample_final),
        }

    print(
        f"Hit@1: {hit_res['Hit@1']:.4f}  "
        f"Hit@5: {hit_res['Hit@5']:.4f}  "
        f"Hit@10: {hit_res['Hit@10']:.4f}  "
        f"Users: {hit_res['users_evaluated']}"
    )

    return {
        "val_auc": auc_score_list,
        "val_f1": f1_score_list,
        "val_precision": precision_score_list,
        "val_recall": recall_score_list,
        "test_auc": auc_score_t_list,
        "test_f1": f1_score_t_list,
        "test_precision": precision_score_t_list,
        "test_recall": recall_t_score_list,
        "hit_per_epoch": hit_hist,
        "hit": hit_res,
        "train_ratio_used": train_ratio,
    }
