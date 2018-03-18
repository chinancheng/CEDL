# Homework3-Policy-Gradient report 
# 鄭欽安 (103061148)
### Problem 1: construct a neural network to represent policy
* Use tf.contrib.layers to construct two layers neural network with one hidden layer.
* This neural network is for generating probability of each discrete action.  
  
```python
fc1 = tf.contrib.layers.fully_connected(inputs=self._observations, num_outputs=hidden_dim, scope='fc1', activation_fn=tf.nn.tanh)
logits = tf.contrib.layers.fully_connected(inputs=fc1, num_outputs=out_dim, scope='fc2', activation_fn=None)
probs = tf.nn.softmax(logits) 
```
### Problem 2: compute the surrogate loss
* Follow this formula  
  
![](http://latex.codecogs.com/gif.latex?L(\\theta)=\\frac{1}{(NT)}(\\sum_{i=1}^N\\sum_{t=0}^Tlog\\pi_\\theta(a_t^i|s_t^i)*R_t^i)) 

   
```python
surr_loss = -tf.reduce_mean(log_prob*self._advantages)
```
###  Problem 3: Use baseline to reduce the variance of our gradient estimate.
  
```python
a = r-b
```
* Loss curve  
<img src=./output_figure/p3_loss.png/>  
  
* Average return curve  
<img src=./output_figure/p3_return.png/> 
  
### Problem 4: Remove the baseline  
  
```python
baseline = None
```
* Loss curve  
<img src=./output_figure/p4_loss.png/>  
  
* Average return curve  
<img src=./output_figure/p4_return.png/> 
  
### Problem 5: Actor-Critic algorithm (with bootstrapping)  
  
```python
def discount_bootstrap(x, discount_rate, b):
  y = [ x[i] + discount_rate*b[i+1] for i in range(len(x)-1)]
  y.append(x[-1])
  return np.array(y)
```

```python
r = util.discount_bootstrap(p["rewards"], self.discount_rate, b)  
a = r-b
```

* Loss curve  
<img src=./output_figure/p5_loss.png/>  
  
* Average return curve  
<img src=./output_figure/p5_return.png/>  
  
### Problem 6: Generalized Advantage Estimation  
```python
a = util.discount(a, self.discount_rate * LAMBDA)
```

* Loss curve  
<img src=./output_figure/p6_loss.png/>  
  
* Average return curve  
<img src=./output_figure/p6_return.png/>  
  
  
