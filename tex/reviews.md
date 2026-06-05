# Peer Review Comments

**Date:** 04-Jun-2026

Mr. Tomas Ortega  
University of California Irvine  
Irvine  
California  
United States  
92697  

**Paper:** JSAC-01286-2025, Decentralized Parameter-Free Online Learning  

Dear Mr. Tomas Ortega,

I am writing to you concerning the above referenced manuscript, which you submitted to the IEEE Journal on Selected Areas in Communications.

Based on the enclosed set of reviews, your manuscript requires a **MAJOR REVISION**.

***

The manuscript studies parameter-free decentralized online learning by combining coin-betting updates with gossip-based information exchange, and it presents an interesting technical direction.

The reviewers however raised substantive concerns that must be resolved before the paper can be considered further. In particular, the manuscript needs to more convincingly establish its suitability for communication-constrained decentralized networks, including a clearer treatment of the communication cost required to obtain the stated guarantees. The revision should also strengthen the theoretical support for the proposed algorithmic variants and clarify the role and advantages of the simplified formulation. In addition, the empirical evaluation should be expanded to include relevant comparisons, accounting for communication cost, and a discussion of performance relative to centralized benchmarks.

I therefore recommend a major revision. The bar for revision is high: the central communication-efficiency and theoretical concerns must be directly and explicitly resolved, and the empirical evidence must be strengthened substantially. A revision that only improves presentation or adds limited experiments would not be sufficient.

***

Your revised manuscript must be submitted to Author Portal, https://ieee.submission.researchexchange.com/journal/jsac-ieee, no later than 6 weeks from the date of this letter, together with a required point-by-point reply that explains how you addressed the reviewers' comments.

If we do not receive your revised manuscript within 6 weeks from the date of this letter, your manuscript will be considered withdrawn.  

If you have any questions regarding the reviews, please contact me. Any other inquiries should be directed to Janine Bruttin.  

================================================

Best regards,

**Dr. Zhiguo Ding**  
Editor-in-Chief, IEEE Journal on Selected Areas in Communications  
zhiguo.ding@ntu.edu.sg  

**Janine Bruttin**  
Administrator, IEEE Journal on Selected Areas in Communications  
janine.bruttin@gmail.com  

---

## Reviewer Comments

### Reviewer: 1

**Recommendation:** Major Revision

#### Comments:
The paper proposes a family of parameter-free decentralized online learning algorithms based on coin-betting and a proof of the sub linearity of the regret. After reading the paper I have the following comments:

*   Although the paper proposes an interesting solution backed by regret guarantees, it is not clear for me what is the considered communication setup. In which way does the proposed approach fit the constrained-network scenario of the call.
*   How is the gossiping happening? What is the communication overhead of the solution? According to the authors $O(t)$ gossiping steps are needed in each round, which means that the gossiping scales with $O(T^2)$… Going back to point 1, how does this impact its application in constrained networks
*   The authors show in Figure 5 the trade-off between communication and performance and claim that „…a linear schedule of the communication allows the decentralized algorithm to closely track the performance of the centralized oracle“. However, from the figure one can see that the losses of the linear schedule are approximately 50% higher than the centralized solution. This is not closely tracking, there is a clear gap that the authors should address
*   Intuition into how everything is tighten to a real application is missing

**Additional Questions:**

---

### Reviewer: 2

**Recommendation:** Reject

#### Comments:
This paper proposes a decentralized parameter-free online learning framework based on coin-betting and gossip mechanisms, claiming to be the first to achieve network regret guarantees without hyperparameter tuning. While the topic is relevant and potentially impactful, I have several fundamental concerns regarding novelty, technical depth, and empirical validation.

First, the claimed novelty is not sufficiently justified. The proposed method essentially combines existing coin-betting techniques with standard gossip-based consensus schemes. While this integration is non-trivial at an implementation level, the paper does not convincingly argue why this constitutes a conceptual breakthrough rather than a straightforward extension of prior work in decentralized online convex optimization and adaptive methods. The positioning against related literature is incomplete and lacks a clear delineation of what is fundamentally new.

Second, the theoretical contribution appears limited in depth. The main regret results largely follow from known reward-regret duality arguments and standard properties of gossip averaging. The decomposition of network regret into local regret plus a disagreement term is expected and has appeared in various forms in prior decentralized optimization literature. The analysis, while technically correct, does not introduce significantly new proof techniques or insights beyond adapting existing frameworks.

Third, there is a notable gap between theory and practice. The theoretical guarantees rely on increasing communication rounds (e.g., $q(t)=O(t)$) to control disagreement, which is impractical in realistic decentralized systems due to communication constraints. However, the experiments are conducted primarily under a constant communication budget, without adequately reconciling this discrepancy. This raises concerns about whether the theoretical results meaningfully explain the empirical performance.

Finally, the experimental evaluation is not sufficiently convincing. The baselines are limited (primarily DOGD and a centralized oracle), and there is no comparison with more advanced adaptive or parameter-free decentralized methods. Moreover, the experiments focus on relatively standard regression tasks and do not demonstrate clear advantages in more challenging or realistic settings (e.g., heterogeneous data, large-scale networks, or adversarial environments).

**Additional Questions:**

---

### Reviewer: 3

**Recommendation:** Reject

#### Comments:

This paper proposes a coin-betting-based decentralized online learning, DECO, for parameter-free updates with regret guarantees. In particular, the paper presents DECO-i and its simplified variant DECO-ii as the main algorithms.

However, the central claims do not yet appear to be fully supported. Although DECO-ii is presented as the main proposed method, the paper lacks sufficient quantitative analysis of the approximation that it introduces relative to DECO-i , as well as its impact on regret over $T$ iterations. In addition, while the paper emphasizes that the method is learning-rate-free, it still depends on the choice of a potential function, which is itself a consequential design decision. As such, the approach does not fully eliminate the need for tuning, but rather shifts it to another component. Moreover, although DECO-ii has a practical advantage over DECO-i by avoidng communication of the wealth variable and only gossiping the accumulated gradient, the paper does not sufficiently justify why the two-variable communication structure in DECO-i should be considered as an appropriate design in the first place. In many decentralized learning frameworks, communication typically involves a single optimization-related state, such as the model parameters or gradient information. In contrast, DECO-i requires transmitting both the accumulated gradient and the wealth variable at each round, which may introduce unnecessary overhead without sufficient justification. More detailed concerns are as follows:

1.  The paper suffers from inconsistent notation, which significantly hinders readability and makes the technical development difficult to follow. For example, the subgradient is denoted by $g$, while $c = -g$ is introduced subsequently. In the regret analysis, both $c$ and $g$ appear interchangeably, which leads to unnecessary confusion. Moreover, the symbol $c$ appearing in Eqs. (57) and (66) no longer represents the coin-flipping outcome introduced earlier, but is instead used with a different meaning. Such overloading of notation substantially reduces the clarity of the paper
2.  The main proposed method is DECO-ii, whereas DECO-i is essentially the standard coin-betting approach augmented with a gossip step, as also acknowledged by the authors. DECO-ii appears to be obtained by replacing $\text{Wealth}_{t-1}$ in DECO-i with the approximation $\text{Wealth}_{t-1} \approx F_{t-1}(G_{t-1})$. The authors should provide a quantitative analysis of the approximation error introduced by this simplification, as well as its impact on regret after $T$ iterations. However, the paper does not provide such an explanation or theoretical characterization. Instead, it only shows experimentally that DECO-ii performs worse than DECO-i, which is insufficient. A rigorous quantitative analysis of the approximation error in DECO-ii and its effect on the regret bound is necessary to substantiate the validity of the method.
3.  Lemma 3 only establishes whether Eq. (4) is satisfied when learning is based on DECO-i. Since DECO-i is essentially corresponds to the conventional coin-betting framework combined with gossip, it is not surprising that the condition holds. However, the paper does not provide a proper analysis to verify whether the same condition is satisfied for DECO-ii, which is the simplified version of DECO-i. Without such an analysis, it remains unclear whether DECO-ii can truly enjoy the same regret guarantees as DECO-i.
4.  The authors claim that DECO has the advantage of not requiring learning rate tuning, unlike conventional online learning methods. However, while DECO removes the need to tune a learning rate, it instead requires storing the cumulative sum of past gradients and choosing a potential function, both of which can significantly affect performance. In other words, although one hyperparameter is removed, another design choice, the potential function, is introduced. As a result, the practical advantage of using this algorithm over existing methods is not sufficiently justified.
5.  The update magnitude in DECO is ultimately determined by $\beta_t$, which itself depends on the cumulative sum of past gradients. Therefore, it is not sufficient to compare DECO only against DOGD, whose learning rate decreases deterministically with $t$. A more appropriate evaluation would include comparisons with methods that also adapt their effective update size based on past gradients, such as AdaGrad, RMSProp, Adam, AdamW, momentum, and Nesterov-type methods applied in the decentralized online learning setting. Without such baselines, the empirical evaluation does not convincingly demonstrate the claimed advantage of DECO.
6.  The discussion of prior works on parameter-free decentralized learning is insufficient, and the experimental baselines are incomplete in this regard. Several studiess on parameter-free decentralized learning already exit (e.g., [1]–[3]). However, the paper neither provide adequate discussion about these works adequately nor includes relevant parameter-free decentralized methods in the baseline comparisons.  
    [1] Li, Jiaxiang, et al. “Problem-parameter-free decentralized nonconvex stochastic optimization.” arXiv preprint arXiv:2402.08821 (2024).  
    [2] Kuruzov, Ilya, Gesualdo Scutari, and Alexander Gasnikov. “Achieving linear convergence with parameter-free algorithms in decentralized optimization.” Advances in Neural Information Processing Systems 37 (2024): 96011–96044.  
    [3] Chen, Xiaokai, et al. “A parameter-free decentralized algorithm for composite convex optimization.” 2025 IEEE 64th Conference on Decision and Control (CDC). IEEE, 2025.
7.  While the paper emphasizes the advantage of being parameter-free, this benefit is restricted to convex online learning settings. In practice, many modern decentralized learning problems of interest are inherently nonconvex. However, the paper does not discuss how this limitation affects the significance of the proposed method. This restricted applicability should be more clearly acknowledged and justified.
8.  While DECO-ii has a practical advantage over DECO-i in that it only gossips the accumulated gradient and no longer requires communicating the wealth variable, the paper does not sufficiently justify why the original communication design in DECO-i should be considered appropriate. In many decentralized learning methods (e.g., [4]-[6]), communication is typically performed over a single optimization-related state, such as the model or gradient. In contrast, DECO-i requires transmitting both the accumulated gradient and the wealth variable at each round. The paper should therefore clarify why this additional communicated variable is necessary and why this design servers as an appropriate reference point for comparison. From this perspective, DECO-ii appears less like a fundamentally new algorithm and more like a reformulation of DECO-i that eliminates the need for two-variable communication by incorporating the role of wealth into the betting-function-based update. In addition, because DECO-i communicates both wealth and accumulated gradients, whereas DECO-ii communicates only the accumulated gradients, the two methods do not operate under the same communication overhead per round. Therefore, the empirical comparison between DECO-i and DECO-ii is not fully fair unless the communication cost is properly normalized. The authors should explicitly justify this communication design and include experiments comparing the methods under the same communication budget, so that the reported performance differences can be attributed to the update rule itself rather than to unequal communication overhead.  
    [4] H. Xing, O. Simeone and S. Bi, "Federated Learning Over Wireless Device-to-Device Networks: Algorithms and Convergence Analysis," in IEEE Journal on Selected Areas in Communications, vol. 39, no. 12, pp. 3723-3741, Dec. 2021.  
    [5] Y. Wang, Y. Xu, Q. Shi and T. -H. Chang, "Quantized Federated Learning Under Transmission Delay and Outage Constraints," in IEEE Journal on Selected Areas in Communications, vol. 40, no. 1, pp. 323-341, Jan. 2022.  
    [6] Z. Zhai, X. Yuan, X. Wang and G. Y. Li, "Decentralized Federated Learning With Distributed Aggregation Weight Optimization," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 48, no. 3, pp. 3899-3910, March 2026.

**Additional Questions:**