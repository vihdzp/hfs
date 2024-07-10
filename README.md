# Hereditarily finite sets

Defines a data type for hereditarily finite sets, and many standard mathematical operations on said type. This can be considered an implementation of ZFC minus the axiom of infinity.

## Introduction

Within [Zermelo–Fraenkel set theory](https://en.wikipedia.org/wiki/Zermelo%E2%80%93Fraenkel_set_theory), the only kinds of mathematical objects are [sets](https://en.wikipedia.org/wiki/Set_theory). For this reason, sets can only contain more sets, which can only contain more sets, and so on. At the bottom of this recursion lies the empty set, the unique set containing nothing.

A set is [hereditarily finite](https://en.wikipedia.org/wiki/Hereditarily_finite_set) when it is finite, and all of its elements are hereditarily finite. The main data type in this crate allows for representation of any hereditarily finite set, within reasonable memory and time constraints.

## Implementation

The most basic type in the crate is `Mset`, which is simply defined as

```rs
struct Mset(Vec<Mset>);
```

This is the type of hereditarily finite [multisets](https://en.wikipedia.org/wiki/Multiset), which are the same as sets but allow duplicate elements. The number of times an element appears in a multiset is known as its "multiplicity". A multiset of multisets can also be viewed as a [rooted tree](https://en.wikipedia.org/wiki/Tree_(graph_theory)#Rooted_tree).

Many basic operations on sets are also implemented for multisets, though they are sometimes defined a bit differently. For instance, the union of two multisets takes each element with its maximum multiplicity among both sets, and the intersection does this for the minimum.

A `Set` is just a wrapper around an `Mset`, with the condition that no two elements are equal, and every element is also a set. This condition is not statically checked, and in fact, it's treated the same as an `unsafe trait`, in that **unsafe code is allowed to assume this invariant**. However, an `Mset` can always be safely turned into a `Set` via `Mset::into_set` or similar functions, at a performance cost.

We also implement `Class`, representing a potentially infinite [class](https://en.wikipedia.org/wiki/Class_(set_theory)). This is just a boxed iterator over `Set`.

## Optimization

19<sup>th</sup> century foundational mathematics were really not optimized for computers. Every non-empty set requires a separate heap allocation, so even moderately large sets might be slow to declare. Determining set equality or membership are nontrivial tasks, and this extends to more complex operations like taking unions or intersections. A naive set comparison algorithm might have exponential complexity or worse.

The [AHU algorithm](https://www.baeldung.com/cs/isomorphic-trees) is what allows us to do any of these operations in a reasonable timespan. The multiple steps within it are implemented as different methods within our internal `Levels` type. The exact invariants and procedures performed by these functions will solidify as more set relations get coded in.

We've strived for each function to be "approximately linear" in the size of the inputs and outputs. The exact complexity is very circumstancial, and in many cases "linear" is a worst-case scenario. For instance, equality between two deeply nested sets will be determined almost instantly if both sets don't have the same cardinality. For shallow and large sets, this might look more like O(n log(n)) instead. The very strict invariants on our types allow us to use `unsafe` code optimizations to eek out a bit more performance.

With all this said, the bottleneck really lies in the mathematics themselves. The [von Neumann natural](https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers#Definition_as_von_Neumann_ordinals) for 15, a number with a measly 4 bits, requires 16,384 allocations to be represented. The number 31 will eat up most if not all of your RAM, and 63 will probably be beyond the reach of computers forever.

For a more extreme example, the [von Neumann hierarchy](https://en.wikipedia.org/wiki/Von_Neumann_universe) divides the universe of sets into layers. Only the set V<sub>6</sub> in this sequence already contains 2<sup>65536</sup> elements; saying that it doesn't fit within this universe is a vast understatement.

In summary: this crate is perfectly usable if you want to test out constructions with small sets, and hopelessly useless otherwise.

## Why?

My main motive in creating this crate was to showcase the absurdity of ZFC. By this, I don't mean that ZFC is bad or even that it's not useful, but rather that it has very silly consequences if you actually try and put it to practice. If you contend that 2 and 3 are sets via [von Neumann](https://en.wikipedia.org/wiki/Set-theoretic_definition_of_natural_numbers#Definition_as_von_Neumann_ordinals), and that ordered pairs are sets via [Kuratowski](https://en.wikipedia.org/wiki/Ordered_pair#Kuratowski's_definition), and that functions are sets of ordered pairs, then you must contend that the set of functions from 2 to 3 is:

```txt
{{{{{}}}, {{{}, {{}}}, {{{}}}}}, {{{{}}}, {{{{}}}}}, {{{{}}}, {{{{}}}, {{{}}, {{}, {{}}}}}}, {{{{}}, {{}, {{}}}}, {{{}, {{}}}, {{{}}}}}, {{{{}}, {{}, {{}}}}, {{{{}}}}}, {{{{}}, {{}, {{}}}}, {{{{}}}, {{{}}, {{}, {{}}}}}}, {{{{}}, {{}, {{}, {{}}}}}, {{{}, {{}}}, {{{}}}}}, {{{{}}, {{}, {{}, {{}}}}}, {{{{}}}}}, {{{{}}, {{}, {{}, {{}}}}}, {{{{}}}, {{{}}, {{}, {{}}}}}}}
```

You can use this crate to directly play around with sets like these, instead of just having them be some mathematical abstraction. The examples might give you guidance for constructions to try out.

Likewise, this codebase serves as a defense of finitism. It's easy to dismiss its entire philosophy as some crackpot theory that doesn't believe in the existence of the set of counting numbers. But really, there's still a lot that can be done within the finite confines of a computer (even if this is really not the best way to go about it).