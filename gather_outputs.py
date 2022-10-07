#Copyright 2022 Google LLC
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *     https://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

import os
import pickle

import numpy as np


def read_pickle_file(pickle_filename, max_eps):

    out = pickle.load(open(pickle_filename, "rb"))
    dist_thress = np.arange(21) * (max_eps / 20)
    ukp_sr, kp_sr = [], []
    mean_dist_ukp, mean_dist_kp = 0, 0
    suc_ukp, suc_kp = 0, 0

    if "idx_success_ukp" in out:
        idx_success_ukp = out["idx_success_ukp"]
        dist_ukp = out["dist_ukp"]
        num_samples = len(idx_success_ukp)
        for dist_thres in dist_thress:
            num_success = (dist_ukp[idx_success_ukp] <= dist_thres).sum().item()
            ukp_sr.append(num_success / num_samples)
        mean_dist_ukp = dist_ukp[idx_success_ukp].mean()
        suc_ukp = idx_success_ukp.float().mean()

    if "idx_success_kp" in out:
        idx_success_kp = out["idx_success_kp"]
        dist_kp = out["dist_kp"]
        num_samples = len(idx_success_kp)
        for dist_thres in dist_thress:
            num_success = (dist_kp[idx_success_kp] <= dist_thres).sum().item()
            kp_sr.append(num_success / num_samples)
        mean_dist_kp = dist_kp[idx_success_kp].mean()
        suc_kp = idx_success_kp.float().mean()

    return ukp_sr, kp_sr, suc_ukp, suc_kp, mean_dist_ukp, mean_dist_kp


# print('ukp_sr: ', ukp_sr)
# print('kp_sr: ', kp_sr)

# # Compute fraction of samples with improvement
# new_success = (~idx_success_ukp & idx_success_kp).sum().item()
# print('New successful attacks from known preprocessing: ', new_success)
# both_success = idx_success_ukp & idx_success_kp
# num_dist_improve = (dist_ukp[both_success] > dist_kp[both_success]).sum().item()
# print(('Percentage of samples that benefit from known preprocessing: '
#       f'{(num_dist_improve + new_success) / num_samples:.4f}'))

# pt_improve = (dist_kp[both_success] - dist_ukp[both_success]) / dist_ukp[both_success]
# print(f'Mean of percentage of distance improvement: {pt_improve.mean().item():.4f}')


max_eps = 50
# last token must be attack algorithm
tokens = []
# tokens = ['resize', '256']
# tokens = ['identity']

files = os.listdir("./results/")
files = [
    f[:-4]
    for f in files
    if os.path.isfile(f"./results/{f}") and f.endswith(".pkl")
]

ukp, kp = {}, {}
output_list = []

for f in files:
    if not all([t in f for t in tokens]):
        continue
    # if ('-tg' in f and not targeted) or ('-tg' not in f and targeted):
    #     continue
    print(f)
    (
        ukp_sr,
        kp_sr,
        suc_ukp,
        suc_kp,
        mean_dist_ukp,
        mean_dist_kp,
    ) = read_pickle_file(f"results/{f}.pkl", max_eps)
    ukp[f] = ukp_sr
    kp[f] = kp_sr
    output_list.append(
        f"{f}: {suc_ukp:.4f}, {mean_dist_ukp:.4f}, -, {suc_kp:.4f}, {mean_dist_kp:.4f}, -"
    )

print("ukp")
for k in sorted(ukp.keys()):
    print(f"'{k}': {ukp[k]},")

print("kp")
for k in sorted(ukp.keys()):
    print(f"'{k}': {kp[k]},")

for l in sorted(output_list):
    print(l)
