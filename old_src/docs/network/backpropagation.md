Okay so this is what we want to do for this function.
We want to calculate the derivative of the cost function
with respect to ANY weight, so that we can generate a weight
gradient.

  $$\frac {\delta c}{\delta w}$$

But because of the fact that any weight isn't **directly**
an input to the cost function $C = (o - e)^2$, we need to
use the chain rule of calculus.

$Z = aw + b$ &nbsp;•&nbsp; _One neurons activation_ <br />
$A = σ(Z)$ &nbsp;•&nbsp; _One neurons activation through an activation function_ <br />
$C = Cost(A, e)$ &nbsp;•&nbsp; _Where e is the expected output for one output neuron_ <br />

Any $w$ affects $Z$ which affects $A$ which affects the cost $C$.
The chain rule of calculus for how any weight influences the cost function can be summed up by this expression: 

$$\frac{\delta C}{\delta w} = \frac{\delta Z}{\delta w} * \frac{\delta A}{\delta Z} * \frac{\delta C}{\delta A}$$

Keep in mind that the cost is calculated in the last layer, therefore that equation is only valid for the last "layer" of weights.

The last expression, $\frac{\delta C}{\delta A}$, is the cost of a single node. Because the mean squared error function is $(o - e)^2$, the derivative of that function is $2(o - e)$. Therefore: $\frac{\delta C}{\delta A} = 2(o - e)$

The second to last expression, $\frac{\delta A}{\delta Z}$ is how the activation changes in change of the activation input, the derivative of our activtation function. We're probably going to be using $ReLU$ for most cases, or even leaky $ReLU$. Therefore: $\frac{\delta A}{\delta Z} = x > 0 \text{ \{ 1 \} else { 0 }}$ for $ReLU$.

The first expression, $\frac{\delta Z}{\delta w}$, is how much the neuron activation pre-$ReLU$ responds to any change to any weight. It's pretty easy as $Z$, or our pre-$ReLU$ function is defined as: $$Z = w_0a_0 + w_1a_1 + w_2a_2 \text{ ...} + b$$
We can se that if we'd like to know how much any weight, like $w_1$ affects $Z$, the answer is just going to be $a_1$ (the neuron connected by that weight from the previous layer) Therefore: $\frac{\delta Z}{\delta w} = a$

## http://neuralnetworksanddeeplearning.com/

$$a^l_j = \sigma \Biggl(\biggl(\displaystyle \sum_{k} x_i w^l_{jk}a^{l-1}_k\biggl) + b^l_j\Biggl) $$
