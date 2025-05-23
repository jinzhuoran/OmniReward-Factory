# Copyright 2024 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np

from ...extras.misc import numpify

if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeAccuracy:
    r"""
    Computes reward accuracy and supports `batch_eval_metrics`.
    """

    def _dump(self) -> Optional[Dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[Dict[str, float]]:
        chosen_scores, rejected_scores = numpify(eval_preds.predictions[0]), numpify(eval_preds.predictions[1])
        if not chosen_scores.shape:
            self.score_dict["accuracy"].append(chosen_scores > rejected_scores)
        else:
            for i in range(len(chosen_scores)):
                self.score_dict["accuracy"].append(chosen_scores[i] > rejected_scores[i])

        if compute_result:
            return self._dump()


def suff_stats(h, m, epsilon):
    """
    +------------+-----------+-----------+-----------+
    | Notation   |          Model Prediction         |
    |            |     <     |     =     |     >     |
    +------------+-----------+-----------+-----------+
    |  Human   < |     C     |     Tm    |     D     |
    |  Label   = |     Th    |    Thm    |     Th    |
    |          > |     D     |     Tm    |     C     |
    +------------+-----------+-----------+-----------+
    C: Consistent on the preference,
    D: Discordant on the preference,
    Th: Human ties but model doesn't,
    Tm: Model ties but human doesn't,
    Thm: Both human and model ties,
    epsilon: threshold for ties
    """
    C = D = Th = Tm = Thm = 0

    for hi, mi in zip(h, m):
        if hi == 0 and abs(mi) <= epsilon:
            Thm += 1
        elif hi == 0:
            Th += 1
        elif abs(mi) <= epsilon:
            Tm += 1
        elif hi * mi > 0:
            C += 1
        else:
            D += 1
    return C, D, Th, Tm, Thm


def calc_acc(C, D, Th, Tm, Thm):
    # This function calculates the current accuracy based on the statistics
    return (C + Thm) / (C + D + Th + Tm + Thm)


def calc_accuracy_with_ties(h, m):
    """
    algorithm: https://arxiv.org/abs/2305.14324
    O(N^2logN)
    Input:
        h: list of N human labels, 1 for prefer A, -1 for prefer B, 0 for ties
        m: list of N model predictions, can be obtained by Score(A) - Score(B)
    Output:
        acc_star: accuracy-with-ties
    """
    try:
        C, D, Th, Tm, Thm = suff_stats(h, m, -1)

        sorted_pairs = sorted(zip(h, m), key=lambda x: abs(x[1]))

        acc_star = float('-inf')
        epsilon_star = 0
        epsilon_curr = -1

        current_stat = {
            'C': C, 'D': D, 'Th': Th, 'Tm': Tm, 'Thm': Thm
        }
        # print(current_thresholds)
        for hi, mi in sorted_pairs:
            # update the statistics by removing the current pair
            if hi == 0 and abs(mi) < epsilon_curr:
                current_stat['Thm'] -= 1
            elif hi == 0:
                current_stat['Th'] -= 1
            elif abs(mi) < epsilon_curr:
                current_stat['Tm'] -= 1
            elif hi * mi > 0:
                current_stat['C'] -= 1
            else:
                current_stat['D'] -= 1

            # update the epsilon value
            epsilon_curr = abs(mi)

            # update the statistics by adding the current pair
            if hi == 0 and abs(mi) <= epsilon_curr:
                current_stat['Thm'] += 1
            elif hi == 0:
                current_stat['Th'] += 1
            elif abs(mi) <= epsilon_curr:
                current_stat['Tm'] += 1
            elif hi * mi > 0:
                current_stat['C'] += 1
            else:
                current_stat['D'] += 1

            # calculate the new tau value
            acc_curr = calc_acc(**current_stat)

            if acc_curr > acc_star:
                acc_star = acc_curr
                epsilon_star = epsilon_curr
            # print(current_thresholds)
        # print("epsilon_star:", epsilon_star)
        return acc_star
    except Exception as e:
        print("Error in tie_calibration:", e)
        return 0


def calc_accuracy_without_ties(h, m):
    """
    Input:
        h: list of N human labels, 1 for prefer A, -1 for prefer B, 0 for ties
        m: list of N model predictions, can be obtained by Score(A) - Score(B)
    Output:
        acc_star: accuracy-without-ties
    """
    C, D, Th, Tm, Thm = suff_stats(h, m, -1)
    return C / (C + D + Tm)

# if __name__ == "__main__":
#     h = [1, -1, 0, 1, -1, 0, 1, -1, 0]
#
#     scores_A = [0.9, -0.7, 0.1, 0.8, -0.6, 0.2, 0.7, -0.5, 0.3]
#     scores_B = [0.1, -0.8, 0.5, 0.5, -0.3, 0.3, 0.4, -0.4, 0.4]
#     m = [a - b for a, b in zip(scores_A, scores_B)]
#
#     print("Accuracy with ties:", calc_accuracy_with_ties(h, m))
#     print("Accuracy without ties:", calc_accuracy_without_ties(h, m))
