## Probability 

1. Sample Space

The sample space (Ω) is the set of all possible outcomes of an experiment or random process.

Example: When rolling a six-sided die, the sample space is Ω = {1, 2, 3, 4, 5, 6}.

2. Random Sample

A random sample is a subset of individuals or observations from a larger population, where each individual has an equal chance of being selected.

3. Events

An event is a subset of the sample space, representing a particular outcome or set of outcomes.

Example: When rolling a die, the event "rolling an even number" is E = {2, 4, 6}.

4. Probability Function

A probability function P assigns a probability to each event in the sample space, satisfying these axioms:
1. 0 ≤ P(A) ≤ 1 for any event A
2. P(Ω) = 1 (the probability of the entire sample space is 1)
3. For mutually exclusive events A and B, P(A ∪ B) = P(A) + P(B)

5. Complement of an Event

The complement of an event A, denoted as A', is the set of all outcomes in the sample space that are not in A.

P(A') = 1 - P(A)

6. Types of Events

a. Joint Events: Two events that occur together.
   P(A and B) = P(A ∩ B)

b. Disjoint Events: Events that cannot occur simultaneously.
   If A and B are disjoint, P(A ∩ B) = 0

c. Dependent Events: The occurrence of one event affects the probability of the other.
   P(B|A) = P(A and B) / P(A)

d. Independent Events: The occurrence of one event does not affect the probability of the other.
   P(A and B) = P(A) * P(B)

Now, let's implement these concepts in Python:



```python
import numpy as np
from itertools import product

# 1. Sample Space
def create_sample_space(experiment):
    if experiment == "die":
        return set(range(1, 7))
    elif experiment == "coin":
        return {"H", "T"}
    else:
        raise ValueError("Unsupported experiment")

# 2. Random Sample
def random_sample(population, sample_size):
    return np.random.choice(population, size=sample_size, replace=False)

# 3. Events
def is_event(outcome, event):
    return outcome in event

# 4. Probability Function
def probability(event, sample_space):
    return len(event) / len(sample_space)

# 5. Complement of an Event
def complement(event, sample_space):
    return sample_space - event

# 6. Types of Events
def joint_probability(event_a, event_b, sample_space):
    joint_event = event_a.intersection(event_b)
    return probability(joint_event, sample_space)

def are_disjoint(event_a, event_b):
    return len(event_a.intersection(event_b)) == 0

def conditional_probability(event_a, event_b, sample_space):
    joint_prob = joint_probability(event_a, event_b, sample_space)
    prob_b = probability(event_b, sample_space)
    return joint_prob / prob_b if prob_b > 0 else 0

def are_independent(event_a, event_b, sample_space):
    joint_prob = joint_probability(event_a, event_b, sample_space)
    prob_a = probability(event_a, sample_space)
    prob_b = probability(event_b, sample_space)
    return np.isclose(joint_prob, prob_a * prob_b)

# Example usage
die_space = create_sample_space("die")
print("Sample Space (Die):", die_space)

event_even = {2, 4, 6}
print("P(Even):", probability(event_even, die_space))

event_gt3 = {4, 5, 6}
print("P(>3):", probability(event_gt3, die_space))

print("Complement of Even:", complement(event_even, die_space))
print("P(Not Even):", probability(complement(event_even, die_space), die_space))

print("Joint P(Even and >3):", joint_probability(event_even, event_gt3, die_space))
print("Are Even and >3 disjoint?", are_disjoint(event_even, event_gt3))
print("P(Even | >3):", conditional_probability(event_even, event_gt3, die_space))
print("Are Even and >3 independent?", are_independent(event_even, event_gt3, die_space))

# Example with coins
coin_space = create_sample_space("coin")
two_coin_space = set(product(coin_space, coin_space))
print("\nSample Space (Two Coins):", two_coin_space)

event_at_least_one_head = {outcome for outcome in two_coin_space if 'H' in outcome}
print("P(At least one head):", probability(event_at_least_one_head, two_coin_space))

```

Let's break down the code and concepts:

1. Sample Space:
   We create sample spaces for different experiments (die roll, coin toss).

2. Random Sample:
   The `random_sample` function uses `np.random.choice` to select a random sample from a population.

3. Events:
   The `is_event` function checks if an outcome belongs to an event.

4. Probability Function:
   The `probability` function calculates the probability of an event as the ratio of favorable outcomes to total outcomes.

5. Complement of an Event:
   The `complement` function finds the complement of an event by subtracting it from the sample space.

6. Types of Events:

   a. Joint Events:
      The `joint_probability` function calculates P(A and B) by finding the intersection of two events and calculating its probability.

   b. Disjoint Events:
      The `are_disjoint` function checks if two events have no common outcomes.

   c. Dependent Events:
      The `conditional_probability` function calculates P(A|B) using the formula P(A and B) / P(B).

   d. Independent Events:
      The `are_independent` function checks if P(A and B) = P(A) * P(B).

Formulas used:

1. Probability: P(A) = |A| / |Ω|
   Where |A| is the number of outcomes in event A, and |Ω| is the total number of outcomes in the sample space.

2. Complement: P(A') = 1 - P(A)

3. Joint Probability: P(A and B) = P(A ∩ B)

4. Conditional Probability: P(A|B) = P(A and B) / P(B)

5. Independence: P(A and B) = P(A) * P(B)

The code demonstrates these concepts using examples with die rolls and coin tosses. For instance:

- We calculate the probability of rolling an even number on a die.
- We find the complement of the "even" event and verify that P(Even) + P(Not Even) = 1.
- We calculate the joint probability of rolling an even number greater than 3.
- We check if "Even" and ">3" are disjoint or independent events.
- For coin tosses, we create a sample space for tossing two coins and calculate the probability of getting at least one head.

