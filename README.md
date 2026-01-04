# UNN
Understanding neural networks

## Preface
Most people here don't know me, as i mainly live on the Ollama Discord server, under the nick "Drazdra". That is the easiest way to contact me. I rarely visit Github (maybe several times a year). 

#### First warning
This is top abstract stuff, if you are not into that, don't go any further. There is no empirical research data or code.

#### Second warning
This field develops too fast, so i might not know about a lot of already existing research papers as i don't read them and don't work on inference engines.

#### Third warning
Forgive me if it's all too simple or naive for you :). The truth is.. that was my goal :).

Content:
1. A very abstract understanding of what neural networks are in general.
2. An overview of the way transformers are made conceptually, easy to understand
3. Several of my own ideas on how to make AGI :).
4. ...

#### Preface.
This text is just an attempt to explain things about neural networks and transformers without any confusing ML slang, without any math, without any code. It doesn't mean it's shallow or simplified, more like the opposite.

Its goal is to make you understand what's actually happening, why, and what for, without having any special knowledge in the field. Transformers are pretty simple, if you understand what they actually do :).

I recommend reading the whole thing before getting back at me :). 

So, if you like neural networks...

# Chapter 1.
## Neural networks and inference: ontological methodology.

I'm sure everyone here knows how neural networks work.. yet please allow me to share my own way of seeing them, as i believe my view provides a clear, intuitive and effective way to see and engineer neural networks. After all, i have been arguing about this for years now :).

Yes, i oversimplify certain things but not to the point of making them wrong *conceptually*.

Actually, the whole point here is moving up from the implementation plane to the conceptual plane and to making it as abstract as possible.

We can have any amount of implementations but *conceptually* all neural networks are still the same :).

Let's start with the most naive question ever, that nearly everyone can answer: how do neural networks differ from regular software? 

To answer that, i suggest seeing a neural network as a "mind" *of a kind* :). But please, don't get triggered here, i don't mean it's a *human* mind :). Although it's certainly a mind as it can *analyze and predict*.

Yes, it's the most primitive type of analysis and it processes data not the way we do, but it can still do both. It can analyze data to detect where the cat is and where the cup is and it can predict words well enough even to write essays, it even creates videos now by predicting the next frames.

To create this mind, we, as humans, code only its brain structure and functions, while the actual content of the mind is structured on its own, automatically, with the help of the structures and functions we have created. 

And, *for now*, it's the same as it is with humans: we just can't inject skills or knowledge right in. We cannot take a person and transfer our existing skills or knowledge right into their mind. 

In fact, we literally just can't teach anybody anything :).

What we *can* do is provide the *conditions (including the information)* for the mind to learn. 

And even then, it may still not end well.. *like solfeggio and me* :).

So what's teaching then? Teaching is merely creating the *conditions* under which one has higher chances to *form structures* that embed new understanding and new skills. 

When you "learn" a foreign language, your brain forms new structures. When you "learn" to play the guitar, your brain forms new patterns for controlling your fingers and for a new type of hearing.

Why is "learn" in quotation marks? For the very same reason! 

Can you even say it's *you* learning it? Not really, you just spend time trying something and then *something* emerges. Or does not emerge :).

You do not patch the connections in your brain manually. They just emerge somehow, if you try hard.

It's the hidden work of the internal implementation of brain structures and functions, which we neither control nor feel. We just get new patterns created in our brain that we can use when they are there.

Same goes for neural networks. They don't control *training*. New patterns just form inside with the help of algorithms we have created for that purpose.

One more time: it's all about just getting new patterns somehow created inside the info space. That's how "learning" happens.

We get the patterns capable of new functions. And we get emerging capabilities.

### But what do these patterns reflect? 

It's easy. Just think about the way models get trained: we feed them with an ordered list of repeatable elements (patterns) and we want them to reproduce something specific as a result - certain wanted patterns.

So, what can they do to achieve that? Of course, they simply have to learn the mutual compatibility of all these elements we show them. How every element fits with every other one, and what the chances are for it to appear there, depending on what we already have in our pattern.

Isn't it simple?

Let's rephrase it: neural networks form patterns that reflect the *chances* for every data element to happen next to any other element.. all depending on other elements around (which we call "the context").

Or, you can say they reflect how closely certain elements (pieces of data) relate to each other, depending on the surrounding elements.

Think about a taste for fashion :). It is not about specific fashion pieces, it's about feeling what goes well with what, depending on the whole ensemble and surroundings. See? Simple.

And that's it. Nothing else. It does NOT store any facts or any information. It doesn't have any texts inside. It doesn't do "cryptography" of your information. There is no text, no images, no videos, no sounds inside. It's not a fancy encoding of your specific information, it doesn't encode any single sequence of data elements.

What it has is merely a huge amount of reflected patterns, embodying a cloud of relation probabilities. 

And that is for all the elements' configurations it has seen - for all patterns.

In other words, neural networks do not reflect the data itself the way storage systems do. Neural networks reflect how data is organized, the possible relative distribution of its elements.

For example:
How closely a kitten is related to "cute" and "milk". How closely snow is related to "cold" and a "blanket". 
And how closely a blanket is related to coziness and coziness to a cat :). 
And that is within the context of the winter and within the context of underwater swimming.

But don't get me wrong, neural networks can go beyond just human ideas or words. They can process any data elements - sounds, syllables, numbers, images, weather, galaxies, anything.

Based on this, let's write down what defines any neural network architecture, what *makes* it one:
 - how it represents (stores) these patterns 
 - how it actually finds/matches these patterns 
 - how it processes these patterns (inference, classification, distillation, etc)
 - how it updates these patterns

If you go down one abstract level and define how all that should be structured and work, you get a specific neural network architecture.

If you go down one more abstract level, you implement these structures in a certain form, like in code or with a physical device, and get a specific model/engine pair.

But at the top level, all of *machine learning is simply creating these patterns*. 

Inference is simply an algorithm that can *continue* the pattern, by finding the best fitting continuation based on all the patterns it knows.

Classification is simply finding the best fitting pattern within the stored ones.

And memory, learning, and thinking are *matching*+*processing*+*updating* the patterns.

Very simple, isn't it? It's all about beautiful patterns :). 

Before moving on, let's devote a bit more attention to the matching part, as it has a trick up its sleeve :).

That trick is in treating the existing context as a specific *point of view* to find the best matching pattern. So that within a single huge pattern we can find/pack a variety of patterns, depending on the perspective we take.

You can visualize it as a starry sky. When a neural network gets trained, it creates a starry sky pattern.
You can see the constellations forming certain specific patterns but if you move to another star system, the same stars now form a different pattern.

Seeing multiple patterns within one single pattern is just a matter of perspective or, in other words, of the context.

Now, let's move on.

### How can it store and process all these probabilities? 

It just uses a hell of a lot of conditional "if/then/else" statements... 

Sorry, i just couldn't resist making this joke :). It's not literally the case, of course. 

Mostly, in modern neural networks, we just use math instead of actual code branching/logical rules, etc. 

The patterns inside are represented as numeric values, so math operations naturally process these patterns and everything just flows natively. By "comparing" these numbers, neural networks can find how similar the patterns are. By "adding up" numbers they can just mingle any patterns together. 

In a way, you can see it as merging two images within your image editor. The image files on the inside are just lists of numbers that represent the brightness of every dot, so we can simply add up the values from both files to get a new image - a mingled image uniting both patterns. 

In a way, neural networks do the same. They just know how to properly merge numbers to get a new pattern or to match them and find how similar they are.

(we still may have certain hard conditions like in ReLU or samplers but more on that later)

Once again:

a) Neural networks usually do not operate with stuff like pure true or false conditions in the code but rather work with floating-point values whose "trueness" is evaluated within the context by "matching similarity". Most of the time, the result of matching just changes the existing state (changing patterns), not choosing between different code blocks. Patterns just merge naturally through mathematical language.

b) There is a huge amount of values in neural networks, which makes it possible to create an immense pattern and to store vast amount of possible *perspectives*, to see any required part of this pattern, to match the best fitting part.
(to repeat once again, it's a pattern of probabilities of what goes well with what)

c) Any value in the "dimensions" can work as one more perspective on the same pattern, it can be used to find a required sub-pattern, serving as a "branching condition" for concept extraction.

This is a very hard thing to grasp (high-dimensional geometry in the semantic space), so i will explain it in a way that anybody can understand :). 

> How can a single value in a single dimension have a drastic decisive effect on a pattern consisting of 10000000 other values? Just imagine a picture of a statue in a textbook. It's a beautiful piece of ancient work, it has tons of dimensions - weight, color, material, age, complex shape, history, art school, author, etc. And now, a bad student takes a pencil and draws on it.. well.. let's say a mustache. The whole thing suddenly changes :). That's how a single new value can affect the entire concept and its perception.

> Or, imagine a great, tasty, nice-looking fruit - it's in the "good food" category. Now just add a worm to it - and the category is all different now :).

d) Neural networks can have multiple separate "big" patterns with sub-patterns.

e) The way neural networks process patterns is non-linear, which is a fancy way to say that one more single tiny dot in a pattern can cause an avalanche and a typhoon at once! So, *in a way*, they *do* have these if/then/else blocks.. *runs away giggling*

Under the hood, we may have scary matrix multiplications, sigmoid functions, normalizations and so on, yet conceptually they all just serve as means to match the patterns and to mingle the patterns :).

So you can change the sigmoid function to something else, you can replace the dot-product with qubits, you can do lots of other things, but conceptually it will be the same - implementing the pattern matching and growing it further according to the rules of our learned pattern :).

### How does it know which patterns are there and which ones match?

Modern training is not really intelligent. Usually, during training, our functions just try to change the values in the neural network's structures until they finally adjust the internal pattern so that it can reproduce the new pattern you gave it.

The check-up here is whether a neural network can match (detect context) and continue the pattern properly, giving you a fitting reply. After all, the reply is just a pattern made of words :).

Under the hood, we usually just use gradient descent algo that tries to find which values to change and how to change them to best reflect your own pattern. 

Getting back to the starry sky example, we can see it as changing the stars' positions so that when viewed from the correct viewpoint (context), we can see only the fitting neighboring stars at the right distances, giving us only the fitting possible constellation that can tell us which star comes next.

Of course, for every *next* desired output we still may change *the same* values we changed for the *previous* desired output.. Which means that whenever the model learns something new, it may distort whatever it has learned before.

And that's exactly why training is lossy by definition. We use the same parameters to reflect multiple patterns, imprinting an immense amount of sub-patterns into a single wooden frame, which can be seen from various angles or in a different light.

And this is basically how the whole training process works.
(I am omitting here various tricks that help models to reproduce the pattern faster than by naively trying all possible combinations)

So, again, neural networks do not store the data we show them, they store only an approximation of the data structure, the internal organization of the data, not the data itself. And to do this, they form patterns..

Which means - a neural network is a *pattern processor* :).

And that's exactly why we can use it for the understanding and development of *anything*, apart from pure Chaos :).

Because understanding something means *being able to predict* it. 

To understand something we first analyze it by trying to grasp the specific patterns of the process and then we correctly predict the development of these very patterns over time, which proves we have grasped them adequately. 

If we cannot predict its continuation, it means we do *not* understand something that goes on - we don't see the whole pattern. 
(or there is no pattern and there is the pure Chaos factor).

That's why there is no difference for neural networks whether it is DNA or sound or text or images or the laws of physics or traffic or human faces or a weather forecast. They work in a way that grasps any patterns in any system in the same way.

### So, what does this understanding give us?

It provides basic concepts for building good AI.
1. It should allow forming the patterns in the fastest, easiest and most flexible manner.
2. It should allow forming the meta-patterns - patterns for the previously formed patterns, with any number of reflection levels.
3. It should allow for rewriting the patterns - updating them.
4. It should be able to match patterns (and find the perspectives).
5. Updates should be minimally functionally degrading for existing patterns.

This is valid for *both* the training part when it creates a reflection of data organization AND the inference part, where it creates a prediction pattern.

Simple, isn't it? You want to have progress? Add dimensions, *add reflection layers*, add update mechanisms.

# Chapter 2: explaining transformers with ponies.
Now, let's do some understanding of what transformers are. We will talk about decoder-only transformers, the ones that we all use now for text generation.

In this chapter, we go down a level of abstraction to talk about a specific neural network architecture. It is going to be a bit less abstract and will provide a bit more detail on actual architectural implementation specifics. But we won't go to the level of actual engines implementing the architecture.

First, i believe the term "transformers" itself might sound fun and is correct in the domain of *mathematical implementation*, but it doesn't really help much to understand what they do.

What i would actually call "transformers" is the _incremental associative morphers_. 

Does it not sound that cool? :) 
Yet this naming is way closer to what they do at a high level and less confusing. To me, at least - and, i believe, soon for you too.

So, let's unpack it a bit.

Transformers have three main conceptual blocks:
1. Input
2. A copy-pasted group of certain layers repeating over and over - the "repeating" blocks
3. Output

These blocks play different roles and we will take a good look at each of them.

### Part 1 - The input block
The role of the input block is to translate your words into the "brain signals" of the model.

The input just translates each element of the text we send into the patterns your model knows. These patterns define the place of the text element you provided within the grand pattern the model has learned. 

It's the internal *interpretation* of our data the model has learned. It allows the model to *relate* your text to everything else it has seen and to find a probable continuation to it, something that can *happen next* in your text.

In text transformers, these input patterns are limited to a *fixed* list after training. So the model has no access to the "raw" sensory data anymore - it can't see anything *new* that it didn't see during training. You can't send a word in an alphabet the model doesn't know. You cannot send an image of handwritten text. It will just have no idea what it means in the grand pattern, where its place is or what goes next. 

Does it mean that the input should *always* be split into fixed pre-learned patterns? 

Nope. For example, for images and sounds it can be different. A brighter image can result in higher values within the input pattern. This way during training the model doesn't have to see every possible combination of brightness to learn how to understand the "brightness". The input becomes variable and the model can still match it because it's still similar to what it has learned. And so it can react to the raw data variations, rather than just having a fixed list of patterns.

#### How do they learn to convert text into.. patterns?
During training, neural networks learn to represent every possible element we send to them as a specific internal pattern, recorded as a long list of numeric values. Just the way image files are stored as numbers on the inside.

Every single "row" of such values is called a *vector*. We call it a vector simply because of the way we deal with its numbers - how we add and compare them. But as this is not a mathematical text, i will just use the word *row*, and for its content i will say values, and for their position in the list, i will use *axes* or *dimensions*.

A table of these is usually called a "matrix" and a single "book" of many matrices is called a "tensor", but i will just say *matrix*.

And *pattern* here is any single numeric representation, be it a vector, a matrix or a multidimensional tensor.

Now that i've triggered the serious ML scientists and they have closed the page, let's go on :).

So we have a text to process and we need to split it into a list of basic elements. 

Could we split it per character? Yes, but in the current architecture it produces so many combinations that it becomes too slow and expensive to use. One token on average equals 3-4 characters. If it were 1 character per token the text generation would slow down by at least 3-4 times. Not to mention the training costs! Just imagine how many more combinations it would create!

Then, could we split it into words? Nope. Words in many languages often inflect, so it is not an efficient way to do it - there are too many of them. Also, we have other things in a text, like numbers.. 

And we decide to do something in-between - to split the text into the most common combinations of characters. That is:
 - Common word parts: hi, under, sta, pro, num, li, etc
 - Static common words: Hi/Hello/hello/HELLO/car/etc
 - Common numbers 1.. 9, 13, 1111, 12345, etc.
 - Punctuation
 - Emojis
 - And so on

Each of these basic data elements is called a token.

Note: i made examples for text-only transformers, but in fact it can be *any* type of data: images (where tokens are typical combinations of pixels), audio (typical frequency combinations), motion (typical coordinate changes) and so on.

### How is this list of tokens created?
Before training, special software takes the entire training dataset and splits it up into a list of basic elements that would provide the smallest text representation. So if our whole training consists of the words "Milk" and "Cat", we get only two tokens: "Milk" and "Cat". But if our training dataset also has "Catie" and "Milkie", our token list would get a third token: "ie". 

On the inside, every token consists of a certain list of numbers, where every number has its position (#1, #2, etc). Every position (column) works as a separate axis or, in other words, as a dimension. Since every token has the same amount of values, all tokens together make a single table (matrix) where every row describes one token. 

If we take all of these tokens of a neural network together, they make *the vocabulary* of a neural network. A list of all elements it knows, even though most of them are not *our* human words but the text chunks.

A numeric list representation of a token is called an "embedding". Further, when i say "token", i usually mean "embedding". But to avoid complicating the story i will just say "tokens". Why? Because "embedding" is just an *internal representation* of a token.

When you send your text, it gets split into these tokens. Then the neural network looks up a tied list of numbers in its vocabulary for every token and just replaces the text with numbers. It puts every token onto a separate row, so instead of a block of text, we get a table/matrix where every row is a single token of our text, represented as a row of numbers.

You can visualize every column (position) of the rows as a separate *axis* of the token's pattern. Like, the first number in the row represents the horizontal coordinate, the second number is the vertical coordinate and so on. 

But it has so many columns! Like thousands! How can we even imagine that?!

That's easy. To imagine more dimensions just think about the pages in a book where each new page carries a sub-pattern of just 1 or 2 axes. Then a single whole pattern is a *whole* book of these sub-pattern pages. So every token is a book of sub-patterns.

How many axes/dimensions/pages do these books have? Just as many as developers decide to give them. It's something they decide upon before training, as the more axes you have, the more time you spend training the model, and the more money you have to pay.

#### What do these columns/axes within the tokens actually mean? How are they used?
And here i could take the easy route and say that every axis has a specific meaning. Like, the first axis represents everything "white", the second axis groups everything "curvy", another one can mark everything related to "names" and so on. And if you have a high value in the first position for a word/token, it would mean it relates strongly to "whiteness", while "blackness" in that same position would have a very low value, showing how far it is from "white". 

But... it wouldn't really be true :). It's actually a *very* misleading way to explain it. And... just wrong... :)

Let's dig into why.

#### What does a neural network learn during training?
It just learns the chance for tokens to happen around each other.

To do that, it has to develop certain ways of *relating* tokens. Of course, some of the tokens come together more *often* than others. So they are *more related*.

How does it record this relatedness? 
Of course, it does so through *similar* patterns of tokens. 

Neural networks do this during training by finding the right values for all token axes and other parameters. The goal is to make certain tokens look like stars and some others like circles :). In a figurative sense... So it's all about figures in patterns, you see :)

And the beauty of it is that it doesn't matter how *big* or *small* these stars are. You still see *stars*, not circles :). You can still match one "star" token to any other "star" token and find the resemblance.

This means that the *actual* numeric values created during training are not *that* important for classification. What's most important are the *proportions* between values, as they form a figure within the pattern. For convenience, we can imagine some proportions of these values as multidimensional "stars", "circles", or even a "pizza without a slice" :).

Once the learning is done, the neural network can then compare these token patterns and see how compatible they are. 

Do not think that every "star"/"pizza"/"circle" pattern is a separate isolated group. No, neural networks have a way to measure how *close* the pattern is to every other token and to decide on its similarity based on that. So we don't really have fixed *groups*, instead we have a way to measure the chance of a token coming based on its *similarity level*. A circle pattern is closer to a pizza pattern than to a star. And each token's pattern has multiple such figures inside.

Until learning happens, nobody knows which token groupings will be reflected, nor which patterns the neural network will create to unite them. But after learning, the figures represent the chances for every token to happen next to every other token, and their *similarity* level forms stable token combinations.

It's like knowing which people will agree to come to a party, depending on who else is there - but for the entire city! :)

In simple words, certain proportions in the token values reflect the fact that tokens a, b, and c have higher chances of coming together, but won't show up if token "d" is there, as its pattern breaks the compatibility.

One more example: say you have tokens with low values on pages 16 and 17 but high values on pages 5, 12, and 121 - this is a sign that the token *might be* *at least* related to a "fairytale"-related token cloud. That's because these tokens always went together within fairytale contexts in the very text the model was trained on, and the model has found a way to reflect it through this "figure" within its patterns. But always remember that the same token can still be related to many other clouds of tokens. Also, a change in any of the axes may suddenly make it related to totally different clouds of tokens, as it will turn existing figures into something new in the same pattern.

The *insight* here is that every token, in fact, is *not* a fixed symbol on the inside. Every token itself is just *a cloud of traits* that can be aligned with *other clouds of traits* - other tokens. Every token is *a concept*.

And yes, any part of the trait can also be a part of other traits, simultaneously encoding multiple clouds of related tokens :). 

So there are no "characters" on the inside, no "words", and no facts of any kind. There is *just* a pattern that can be related to other patterns. 

How do most people interpret this? 
"If the words are linked sensibly, it must mean it's has ideas! It stores facts! It has knowledge!" :)

Sounds fun, but... what it stores are just the chances for data to be organized in certain ways :). 

But i said it's a concept... Concepts? Ideas? What is a concept, even?

"Concept" means a combination of basic elements into a single new basic compound element. A concept car is a car built from a new set of basic components: an electric engine, propellers, carbon, a steam machine, AI, and paws, etc. In our case we are speaking of thinking, so it's about data elements, and for language, it's words. When we *think in language*, we operate with meaningful semantic units. We can build concepts based on functional understanding, traits, or abstractions, all expressed *in words*. The concept of "comfort" includes a lot of basic ideas, where each idea is expressed by a certain word: warm, convenient, safe, relaxing, pleasurable, etc. And we can think about these *words* as its compound elements - basic concepts forming a new concept.

But transformers form *concepts* based merely on the grouping of symbol sequences! So, for them, the concept of "comfort" might be additionally tied to the "concepts" of "like ", "ed", "ing", "car", "dis" and even the comma ",". 

They don't have *any* experience other than text... They make concepts out of the grouping statistics of chunks left by the tokenizer.

*Transformers *lack* the entire necessary layer that makes humans human - language representation.*

We don't just relate the *semantic units* in neural networks, we relate both: meaningless chunks and the proper words that carry the meaning as we mean it :). 

That is, transformers at the raw level often do not manipulate concepts made of *semantically meaningful ideas*, they work with *concepts* made of tokens which may reflect the mere statistical distribution of the *characters*, rather than human *ideas*. It's like humans *partially* trying to form sentences based on most common *sounds* pronounced together, not just on *words*.

That's why tokens are a paradox: on the one hand, neural networks have to use them to create the clouds of related tokens - to "conceptualize" the data. But on the other hand these tokens are often *not* meaningful as we mean it...

...and something tells me, the person who invented tokenization was a programmer :). It perfectly *optimizes* language encoding while simultaneously *muddling up* the whole conceptual purpose of doing it :).

> *..or maybe even a mathematician?*

> But more on the tokenization solution later :).

Even so, neural models manage the trick of imitating human-like *concepts*. How?! Why? There are three reasons:
 - tokenizers also contain many full words and word roots which are *our* concepts
 - the attention implementation doesn't process learnt patterns *separately* but rather always mingles these into a single large pattern, working as a *semantic assembler*
 - the amount of statistical data is immense, allowing the model to learn to translate its conceptualization into ours (remember, transformers started as *translation* neural networks)

These three things partially patch the initial "flaw".

Now that we see how "concepts" emerge in transformers (as typical trait clouds - tokens), let's take a look at the classic example of the "King" and "Queen" tokens. 

There is a talk that if we apply the pattern difference between "King" and "Queen" to the "Man" token, we somehow get the "Woman" token. Or vice versa. Leading to the conclusion that this specific pattern change (vector shift) encodes "gender" information :).

However, it's not. It's because the "Man" pattern is *already* associated with a certain cloud of related ideas (man, dirty, socks) and has a specific shape. By introducing these specific pattern *changes* *to that specific* shape, we make it partially closer to that *other* "Queen" cloud (woman, pretty, stockings) and more distant from certain tokens within its original cloud (man, stinky). 

It may seem like we encode "gender" in these changed proportions, but in fact, we just change the strength of the ties to *multiple* other tokens by modifying the existing pattern. And weren't these patterns already *similar* or... *compatible*? :) And the *change* we introduce may, in fact, bring way more than just gender. Potentially, it can change the relatedness to *all* tokens. 

More than that, if we apply the same change to an irrelevant token (with a very different pattern), it may not introduce anything gender-specific at all because the resulting figure will still be very different. It would just move the token closer to a different cloud of related tokens
 ...*and who knows what dragons live there!*

Summarizing it all, this is how training creates patterns that capture the tokens' mutual relatedness within their probable combinations. 

Once training is complete, all these token patterns are "frozen", and the neural network only reads this learned vocabulary but never changes it. Whatever text we input, it immediately *knows* which tokens go well together next and which do not. It can compare, it can match, it can mingle - all because the text elements are just patterns written in one internal "language". 

And for the same reason, it can easily translate back into human words any *new* mingled pattern. It just finds the most similar pattern in the vocabulary and looks up the associated token. Then it just takes the textual chunk of that token and prints it back for us. And yes, note the "*A*", as there are always multiple similar patterns/tokens :). But more about that later.

One last thing: what if we send the neural network a long text consisting of random characters? No template, just that.

In that case, our model will just *continue* this list of loosely related, meaningless garbage, sticking to the weird pattern we have made. Because the mingled pattern will be extremely noisy and won't contain any figures the model can detect. It won't align with any sensible pattern. It will match only itself, and the continuation will reflect its own configuration, rather than the learned probable patterns.

Until.. it finally stumbles upon a familiar pattern. And then, it will simply continue that pattern, forming sensible sentences. 

For example, if you end your long garbage string with a single question mark, that alone might be enough to make the model continue with a sensible text. That is because:
 - garbage = "weird pattern not matching anything from the common sensible patterns, not even other garbage, as all garbage examples look different. It matches only its own parts"
 - ? = "a known pattern related to a typical, normal reply"
 - "garbage+?" = "tokens typically coming in a reply when nothing else familiar is detected". 

If you send a typical *templated* request with garbage, the neural network will generate a sensible reply right away, because the template tokens act in the same way as "?". The model was already trained on how to reply when the only detectable pattern is the template pattern amidst some noise.

And upon this, we finish the "input" block. 

I think this part was pretty easy, wasn't it? :).

#### Part 2 - the repeating blocks
This is the core of the Transformer, so it will be the longest section, with loads of sub-parts. Let's start.

It consists of a repeating structure where each repeated block comprises the same set of layers, yet with different content. As if somebody copy-pasted one block many times before training. And while they are structurally the same, they all now hold different patterns.

The way they work is like a one-way conveyor where each next block does its specific changes to the pattern.

Each of these repeated blocks of layers is "isolated" from the others, they do not cross-talk in any way; they process data consecutively, block by block.

We can run this whole conveyor many times, each time generating one more new pattern/token, adding it to the original text, making the text pattern longer and longer.

We call this process inference: creating a new pattern (token) from the existing pattern (tokens). 

In a way, you can see the work of a Transformer as a line of people. 

1. The first person splits your request into chunks and writes it down on a piece of paper using an internal pattern language. Then, they "add" a space for the new part of the pattern.

2. The next person in line (a repeating block) interprets the input pattern and mingles their summary into the new part. The original text also changes as the person also mingles in their own new interpretations. Then the person passes the paper to the next person, and so on.

3. The last person takes this paper and translates only that very last pattern part they've created back into human language. They add it to the original text and... feed the paper back to the first person, to be translated back into patterns...

4. The only thing retained between these cycles is the text interpretation, so from the second cycle people don't have to re-interpret the existing text. They reuse their own past interpretation and work only on the newly added part and the one they need to create. That's called the KV cache.

5. The whole operation repeats from step 1. 

I mean, even if you tried, it would be hard to make things any weirder. Doesn't it remind you of the telephone game? :)

They were trained together to produce a certain result, but they never talk to each other, they can't go back, they do not have a separate scratchpad for calculations or "thinking", etc. They have no plan. They just mingle in the very first associations they have with the entire text they have by now, on a token-by-token basis.

#### But how exactly can they interpret the data? 

Spoiler: the whole approach to the implementation feels a lot like, "let's just put several more similar blocks and see what happens" :).

Let's talk about the internal parts of the repeating blocks now. Each one consists of two main things:
 1. The attention block.

    It has a single O matrix and multiple identical sub-blocks consisting of their own Q, K, and V matrices. Each of these sub-blocks works independently, in parallel on the same input, and is called "Attention head".

    The V matrix in each attention head learns a rough idea of how to extract useful data from token patterns, so that their traits can be mingled together in a unique way, without introducing too much distortion. In other words, it determines how to interpret tokens.
    
    During the training, the Q and K matrices learn to decide upon the compatibility of the tokens within the attention head, which results in measuring the "relatedness" of tokens to one another. 
    
    The O matrix sits above these and learns how to extract useful data from the *united results* of all attention heads. In other words, it knows how to create a new single pattern from separate, smaller patterns produced by individual attention heads. This way, after the whole attention block, we have a single mingled representation of data. It also tries to filter out the noise produced by attention heads.

    The entire attention block produces an *overlay* pattern that is blended later into the original token once the attention step is complete. This overlay is a *change* to our old pattern and doesn't have to contain all of the information required for a full token pattern.
 
 2. The FFN block, which goes after the attention block.
    
    This block learns to "*fix*" things after the attention results are added to the original token :). It works like typization in a way, with the difference that it produces the *changes* to be applied to the token as it comes from the attention block.

    So it's like typization *to a degree* of the new token's pattern, that can introduce new traits or dampen existing ones, making it match the fitting typical token clouds better.

    While attention infuses traits from past tokens into every token, this block makes each separate token more *identifiable*.

#### Starting with the attention block: Q and K.

As explained above, at the start, we split the data into tokens and translate them with the input block.

So, here we have the entire context represented as a table where each row is a token, and we start by sending it to the first repeating block's attention. Literally, the whole table goes to each of the attention heads in there.

Each of the attention heads has its own single Q, K, and V matrices it has learned during the training. Basically, that's what an attention head is: the Q, K, and V matrices and some logic for using them.

The goal of attention is to find all *related* tokens in our context and to fuse their traits. This allows us to find something that is related *to all of them* together - that is to their context. In theory, we could just try to combine traits from *all* of the tokens together. But this is impossible, as our token's pattern has a fixed length and can't encompass infinite information. Trying to add everything into one little thing would just ruin its internal structure, it would be a cacophony.

So we need to find only the tokens that actually often come together and so are related. Then we could unite their traits and see what else usually comes together - what else is from that cloud.

These Q and K during the training learn to *detect* certain specific traits that the model sees as stable and relevant figures. So each attention head gets its own specialization - reacting to the specific cloud(s) of tokens.

But more than that, both Q and K learn to express the result of their findings in a *compatible way*, so their results could be *compared*. 

This makes them two parts of the same function, as only the *comparison* of their results is used in the system.

And the goal here is to find out which of the tokens are compatible, so we can mingle them, and how much we can mingle them without breaking the proportions.

Which tokens do we actually compare?
Well, of course all of them. We need to know which tokens the attention head can mingle, so we just go over every pair of tokens - comparing each next token to every preceding token.. *and with itself too*. 

So the word chunk #2 is tested against #1 and.. #2, chunk #3 is tested against #1, #2, and... #3, and so on, for every pair.

> By the way, in ML slang this is called "causal" attention, as in "cause and effect", causal here is literal - "preceding/determining the future". And all of that is done in batches, doing math simultaneously, to utilize the full power of gpu parallel processing

Why do we need to compare a token to *itself*? 
 - The traits we extract *from* the token might break the proportions of the whole token when added back (remember, in the end we mingle everything back into the original tokens).
 - We need to know how much to scale the traits of *all* mingled tokens relative to each other.

As the final purpose of comparison is the mingling of tokens, in every pair we have a token we mingle into and the token we extract the traits from. Let's call them:
 - The token we mingle into: a target token
 - The token we extract the traits from: a donor token.

So, how exactly do we compare them?

The first thing to understand is that we use the "input" version of every token in the sense of "as it came" to the attention block, *before* we changed it here. So the first repeating block's Q/K see the tokens as they are from the input vocabulary, and every next repeating block's q/k already see the tokens after they were processed by all preceding blocks.

The second thing to understand is that before comparing the tokens, every attention head goes over every token and creates its own Q, K, and V *results*/vectors. These are created by multiplying the token content by the content of the attention head's Q matrix, K matrix, and V matrix. So we get three new different patterns/rows/vectors for each token, for each attention head. And each of these patterns is just a single row of numbers.
> I will explain "multiplication" later in the FFN block; this section is too dense as it is

The attention heads care only about *their own* token "interpretations" - their own created rows - they do not interchange their data at all.

So now, when we have this extra data per token, we can actually compare the tokens by these created patterns. 

Specifically for the comparison, we need the patterns created by the Q and K matrices. But... which ones do we use to compare?

Whenever we compare two tokens, we always take:
 - Q row from the target token
 - K row from the donor token.

What do these rows actually *mean* conceptually? My own interpretation of this process is:
 - Q detects if the target token is compatible with the extracted traits *this attention head typically produces*
 - K detects if the donor has traits compatible for extraction *by this attention head*  
 - And Q+K together learn how much to *scale* the tokens' extracted traits, relative to the target token

So, if the tokens are not compatible in some way, the extracted traits are just scaled to near zero and ignored.

Which *traits* does every attention head actually detect? We don't really know. It just automatically happens during the training. The neural network just has to develop some way to relate the most common token clouds by some traits and this makes attention heads specialize on some of these. 

But let me make myself clear here: i believe that Q and K don't just project the two tokens *to compare them through the tokens' "distilled" traits*. It's not about the intersection of *tokens*.

Q and K project a "validation" pattern showing if the token is compatible with the *attention head*. It's like a green lamp showing this token *can* be processed with *this* attention head. Q lights the green lamp if the token can *accept* traits that the attention head usually extracts. And the K green lamp lights if the token *has* traits that this attention head extracts. So they are "compared" against the *attention head* first, and then the results of these two detections are what we actually compare.

If attention head could detect both:
 - compatibility of the target token with the common extracted traits by this head
 - compatible traits in the donor token with this head's extraction

these matrices learn to produce *similarly shaped* results - a green light.

So their similar *shape* expresses their compatibility, while the shapes' combined *size* (the brightness) expresses the *proportions* that the mingled-in patterns should take.

> The shape here means that their comparison produces a positive number, while the size/magnitude is a measure of how large this number is. We will talk about it soon.

How can it even work? Well, the thing is, that these two matrices are all tied to the work of the third matrix - the V one. The one that actually learns how to extract certain traits from the tokens. And so our "green light" is the emergent feature of all *three matrices trained together*: Q, K and V. They learn to be *compatible* with each other through being *trained together*. That's how they develop their specialization.

But... why do we need *two* matrices to detect the compatibility of a token with the V matrix? Why couldn't we detect it with just one matrix?

The thing is, Q and K can perfectly look for *different traits* in the tokens, because we may mingle *different* clouds of tokens here. Yet, these matrices produce results (green light) that should be *similar* for mingling to happen, regardless of *what* they *look at*. 

> Being compatible with traits *as a target token* doesn't mean having donor traits and vice versa. Even more, traits extracted from token A can fit token B fine, but extracted traits from token B might *not* fit token A. This thing is not symmetrical, and we look for *different compatibilities* here. 

And considering that they look for potentially different things, yet their results should match in comparison, the matrices just *have* to abstract their findings into a different resulting pattern language *common to both matrices* - the "green lamp". 

In that language they need to retell two things:
 - whether or not they succeeded in finding the right traits in the tokens
 - how to scale the traits to avoid distortion

Suddenly, the relatedness of the tokens becomes a level of compatibility with the attention head's "preferences" :).

And here you can exhale, as this is the whole concept of "relatedness" :). I've turned everything upside down here to explain, but i think it worked well :).

Now, let's go into more gory details and see *how* exactly we compare. You may safely skip this section.

As i said above, after comparing the Q and K resulting rows we end up with a *single number*. Yep, it all goes down to just one numeric value that we use to scale the traits (the V results).

But how exactly do we get this single number? 

We do this using a method called *dot-product* comparison. I wonder who comes up with these names... So, to do it we just:
 - Place the resulting Q row of numbers over the resulting K row.
 - Multiply the overlapping numbers of these two "layers".
 - Sum all those results up to get a total value.

In a pattern-based way of thinking, we just multiply the symmetrical rays of the two patterns and then add up their final lengths.

If the value we get is high, it means that patterns are compatible - the pairs of their rays mostly were pointing the same way - either both positive or both negative. If not, it means this head can't merge them. Here are 3 examples of "two-axed" patterns: 

```
   Q1   K1     Q2  K2   Result
 - 2  * 2   +  1 * 3  = 7  = match (green)
 - 2  * -2  +  1 * 3  = -1 = no match (red)
 - -2 * -2  +  1 * 3  = 7  = match (green)
```
Then we turn everything below zero into a very small positive value. This way, after scaling, any traits of a donor token that produced a negative result become miniscule and don't change anything in the amalgamation.

Everything positive is first proportionally reduced (to get smaller numbers) and then normalized to fit the range between 0 and 1, so our comparison can only scale the traits down. 

Basically, that's all. That is how we compare the shapes (the signs of all axes) and the magnitude (the final numeric value).

> The question is - do the Q or K matrices *always* produce more or less the same shape when detecting good compatibility? Or might they produce different (per pair) yet similar (within the pair) shapes for compatible token pairs? I did some web search and, although i didn't find any specific research data, it seems that yes, it's more correct than not :). And i believe it's partly tied to RoPE that we will discuss later. But it's just a side interest that doesn't change much conceptually.

#### Unnecessary details about the actual comparison process that you can skip :)
 
Once we have summed up the results, we do a few more operations. These do not change the meaning conceptually, but they make the results more stable, convenient and easier to compare across all tokens.

First, we just scale down the final value by the number of axes there were: we divide the result by the square root of the total number of axes. We do it to avoid turning most values into ~zeros after the softmax. 

Second, we normalize it with the softmax, which means we take the comparison results *of all donor tokens and our target token*, and use these to rank each score against them all. Here we lose the original proportions of the scores, as this function is exponential.

How do we calculate softmax? 
To calculate the softmax, for every token pair we take Euler's number (the constant 2.71828) as a base and raise it to the power of the "reduced" comparison score. For example, if the score is 6, we do 2.71828^6. 

Then, to avoid big numbers, we divide every "powered" result by the sum of all the powered scores - effectively converting every value into a "percentage". And that's how we get our final softmaxed value in the range of 0 to 1.

The effect is that the highest numbers become much, much higher than anything even slightly smaller. It's like every little addition to a high value lifts that score more and more, faster and faster. And negative values become extremely small fractions.

So only the *most* compatible ones can really add to the amalgamation. Even the slightly less compatible ones are reduced to a very small effect. We even have to scale them down first to avoid the total disappearance of any gradient.

#### End of the unnecessary details :)


#### A moment of critique

When we do the comparison with dot product, if a single column/axis is MUCH higher than the others and matches its paired column in sign (both positive or negative), it might cause the final result to have a high value *even* if all of the other columns have different signs but low values. This is a flaw of the comparison method (dot product), and the model *has* to learn to work around it by finding combinations of parameters in the Q and K matrices that make this situation very improbable. 

Theoretically, a model could abuse certain specific axes in the underlying token values for easy detection, by giving them high values as a trait. But these attempts, i believe, are wasted by the later normalization block, so the model mostly has to abuse the Q/K matrices in this regard.

Another thing is that the only way to encode the level of pattern incompatibility here is to use opposite signs for the values or zero values. Just different non-zero values of the same sign can only define the level of compatibility.

All of this is a pretty limited method for encoding a signal. My idea is that it works mostly because all it has to do is detect compatibility, which is a very simple signal to encode. And, it actually *should be* a redundant signal.. but more about that later :).

The Q/K thing looks *very* smart, probably the smartest thing in Transformers, being also very fast in terms of compute. Yes, a better comparison would be not just compatibility with the head but between *actual traits*, but that would be *a totally different attention story*.

#### End of the critique

#### A moment of thinking

Here we get an interesting side conclusion: Q/K may learn not just the level of token compatibility (the critically low score category vs everything else) but also learn to work as a *patcher* to change the size of the patterns, so that the amalgamation of their work ends up in the right proportions. This is interesting as it then learns to encompass two different functions: detecting token relatedness/compatibility and patching the size of mingled patterns for better mingling. Of course, at that, the mingled traits do not have to be *the same* between tokens, as the attention head may extract different ones from different tokens. But considering the amount of other things it should take into consideration, i would guess it's a rather weak emerging feature.

#### End of the thinking


#### Distance between mingled tokens
If you have an inquiring mind, you may have spotted that while we know now how compatible our tokens are, we still have no way of knowing how much they should affect each other, as older tokens should affect new tokens less than fresh ones. The "white" word should not affect all further words in the text as much as the very next one. 

How do we go about it? 
Well, to deal with it, transformers introduce the RoPE trick. 

You can imagine it as an odometer - a device in cars that counts how far the car has been driven. It shows a number in miles or kilometers, where every digit is on a separate rotating disk. Once the odometer reaches "0009", it turns into "0010", and so on. And remember, all disks move *simultaneously* so it's not a sudden shift to +1 in the third disk. No, that disk is slowly crawling there the whole time and could separately be read as being "halfway there" when we have a "5" on the last disk. 

This is exactly how RoPE works. It simply takes the position number of every token (mileage within the context) and converts it into pattern distortions in a *predictable way*. Literally... it pretends that the axes are these disks and rotates them... 

How can it rotate an axis? Well, it can't rotate one axis :). 
Instead, it:
 - Takes a piece of lined paper and puts a dot on it, calling it the (0,0) coordinates.
 - Takes the values of *two* axes, pretends these are (x,y) coordinates and puts a second dot there.
 - Takes a drawing compass, puts it at the (0,0) dot, and draws a circle through the second dot.
 - And then, with a wicked smile, gradually changes these coordinates as if the second dot is alive and crawls upon that circle, until it returns to the same spot.
 - And so on again and again.

It doesn't stop there; it uses separate pairs of axes for various distances. So, if the first pair is for, say, 10 miles (tokens), the next one can be for 20 tokens, the next one for 40, and so on. In this setup, at token #40, the first pair of axes has made 4 full circles, while the last one has made only 1.

But... axes of what? Which pattern do we torture this way? Well, these rotations are applied to the Q and K results.

As i said above, the rotation angle depends on the absolute position of the token in the context - its "mileage" from the start of the context. The greater the distance between tokens, the more axes pairs get rotated.

If the original shapes were *similar*, once we increase the distance between tokens, RoPE rotates their axes, and the shapes lose their similarity and become *less* compatible. 

Can this rotation make different non-compatible shapes falsely compatible? 

Of course, when we increase the distance between tokens, the values in a few axes may suddenly get *more similar* by accident. But usually, it's not enough to make the *whole shapes* similar enough because there are plenty of other axis pairs that are still different. So, an accidental "compatibility" in just one pair of axes after rotation has very little chance of making those two full shapes suddenly falsely compatible.

So, is the RoPE perfect? Sigh. Let's see..

Once we do 180 degrees rotation due to distance between our tokens, vectors suddenly start *restoring* their original similarity, as if becoming more and more compatible, tho the distance just *grows*. 

This is pretty fun, however don't forget that all the "disks" rotate, which means that when our first disk starts to restore similarity, our second disk is already halfway "distorted". So, the overall compatibility is still diminished.

Another thing about RoPE is that, due to the implementation, as our absolute positions of tokens in context become large, the rounding effects in the code may lead to precision loss, digital noise :). So in a real-world engine, the actual precision loss might intervene as we go deeper into the context, due to quantization stuff. If it were linear logic, it wouldn't matter much, but in our case we have non-linear effects where a little change might lead to big consequences, so this thing *may* at times affect results a lot.

A single attention head *may* lie about the tokens' "compatibility" score based on *where* the tokens are in the context, even if their mutual distance didn't change. And it can also lie if the distance between tokens is out of its "safe" range.

And here we get another problem! The longer the context we have, the larger the number of disks that might lie about the compatibility. At the start, only the fastest disks go outside the safe zone quickly, but deeper in the context, a lot of medium-speed disks might also show false incompatibility.

So, let's try to predict some consequences of this fun, thanks to RoPE:
 - Every attention head is applicable only within a certain distance between tokens; otherwise, it lies.
 - At certain relative positions of tokens in the context, an attention head lies.
 - To be able to still understand where there is noise and where there is a signal, the model has to *duplicate* attention heads with the same specialization *for various distance ranges*. So, multiple heads catch the same traits just over different distance ranges.
 - This redundancy is the only way for the neural network later to understand which attention head lies and which ones are telling the truth.

Now consider that RoPE, once it is applied deep in the context, actually warps the original shapes produced by Q and K a lot. If these shapes actually carried some fragile, single-structure signal, there would be a really narrow range of distances where they could still keep both the signal and their similarity.

And... we would hit the need for incredible redundancy in the attention heads, right? The training also would grow a lot compared to non-RoPE solutions.

However... it doesn't happen. Things don't change *that* much. 

Why? Because the signal is very simple: it's not about matching the profile of specific traits between two tokens; it's just about compatibility with the head. 

And this one can be expressed easily in just a fraction of the axes and... it can even be easily redundant across *different distance-range axes/disks*. This makes it easy to develop a pattern that still stays compatible by just reusing abstract, similar patterns over various disks that average each other out. So, when some go out of scope, others just become *more* compatible. And the compatibility signal survives. 

Of course, it can't compensate for any range of distances, as the starting positions of the disks should be neutral. If they were originally opposite, a fast-pair match wouldn't be decisive for the whole Q/K comparison.

RoPE still forces models to create redundant heads for long and short distances to fight false signals when multiple "disks" go out of phase and start to lie. But because of the "compatibility" trick, it's much less redundancy than there would be if Q/K carried a really complex signal about the actual token traits.

Why do we still use RoPE then?

Well, we could send the numeric position of the tokens, but it doesn't work well because the model gets used to certain tokens playing specific roles at fixed positions, and we want it to be as abstract as possible. We don't want the first produced token to always be "hi" just because it's the first one :). Also, this way it's much easier to *scale* the supported context length, as the model learns to encode any distance, rather than just a fixed range of positions. Instead, it learns to translate distance into a compatibility change, even if it makes mistakes.

#### Now let's talk about the *user* side of the story

I hope you've skipped the previous explanation of the internal mechanics, and now we can just speak about the human side of the story :).

To make it easier to understand, let's pretend for a second that by pure random chance our network's attention head has learned to match the traits of a specific idea of humans.. (it can't happen as transformers never create token clouds around human ideas; they are based on token *sequences*, but let's pretend).

Let's say our attention head doesn't care if the tokens are "color-related" or "grammar-related". It only cares if the tokens are "politically" related! (Sigh) So, it's compatible only with tokens that have figures common to politically related tokens. Maybe the head was trained in 2025, who knows?

Then, on a higher level of abstraction, we can believe it measures:
 - The relatedness of the token's interpretation (both are about politics and close enough).
 - The significance of the related part within the whole token (how much is it about politics?).
 - The significance of the related part to the head's focus (is the head about politics?).

How we may think about attention heads then:
 - every attention head has its own interpretation of the tokens
 - when we test tokens for relatedness, we search for it within the context of this interpretation
 - for testing we use the original token values *before* they are interpreted (as they entered this block),
 - as this way we also learn to see how significant the related part is within each token

But... all of that is only in our own heads :). 

It's not what it seems. In a neural network, these are just potentially emerging features that can sometimes happen. But all it does is test the tokens for their compatibility level with the head's learned patterns. And these patterns are based upon... mere sequences of word chunks.

Now, let's take a look at some real-world examples. And sorry, it's not going to be political :).

If our text is "white hare", Q/K checks if they both are compatible with the head and can be mingled, and how much. That is, it determines if they are "related".

If it believes they are, the neural network mingles some traits of the "white" and "hare" tokens, making the result *closer* to specific token clouds. Let's say one of these clouds is a "snow-related" one. 

What *exactly* does our attention head specialize in? Maybe in any adjectives that change nouns? Or just in "snow" token cloud traits? Or something else? Who knows...

But our sum of "white+hare" suddenly is now related to the "snow" token, as:
 - The "white" token often happens in the context of snow, and has traits of the "snow" token cloud;
 - The "hare" token happens in the context of "snow" and has these traits too.

And since our attention head liked these traits, the snow-related cloud of tokens becomes very much related to our new "white+hare" amalgamation :). We just made these traits much stronger. Of course, various attention heads also mingled tons of *other traits*, but at the moment, we only care about this one.

Let's test it with Gemma2-9b (gemma-2-9b-it-Q4_K_M.gguf) (without a template) using standard settings in llama.cpp.

And guess what? We do not get "snow" as our next token *anywhere* in the probabilities. Why? Simply because this token combination doesn't *ever* end in "snow" in actual texts. A "White hare snow" sequence is highly improbable.

What is probable? Judging by the output, the most probable continuations are:
 - comma (,);
 - new line (\n);
 - or the word "is".
 
So, it wants to say either: "White hare," or "White hare is", or "White hare\n". 

That is exactly because of those *other traits* that other attention heads mingled in. 

Those traits were much more prominent than "snow", because these words way more often end up followed by a comma or an english auxiliary verb. And that's exactly how it works. Not by abstract "idea" closeness, but by the *statistical* distribution. So, even a new line right after two words becomes a much more probable option than the "snow", which is nowhere to be found. Weird as it is, traits of the "comma" were already embedded in "white" and "hare". And they were more refined - stronger than the "snow" traits :).

But what about "snow"? Are the traits of that cloud even there, as i explained, or was i wrong? Are they encoded in the token patterns? To test this, let's nudge our neural network to produce a *surface* token. To achieve that, we just add the word "upon". "Upon" is almost always followed by some surface in the text, so it should work well. But which surface will we get? Grass? Floor? Asphalt? Clouds? Let's see:

White hare upon -> ```the snow,
A flash of white, a silent flow.
Across the field, a fleeting sight,``` (seed: 2366235071)

Ah, i forgot about "the"! But it's not important here, as you can see, the "snow" became a very probable continuation.

Now, let's change it to "Brown"..

Brown hare upon -> ```the moorland,
A flash of white against the green,```

And as you can see, if we change it to "brown", we get the "moorland" as the surface instead. 

So, the "snow" traits are encoded in the "white hare" *even* when the next token is *not* snow.

...but the fun thing is that even for "brown hare", we still get "white" as well. How come? Our hare is brown! 

Well, it's just because the word "hare" often happens around "white" in texts, just as it does with "brown". So, we still got "white" traits in the mingled pattern.

The model wasn't trained enough to grasp that having "brown" before "hare" makes "white" appearing *later* in this combination very improbable. There are no separate, strong enough "brown hare" traits with their own separate token cloud, so to say.

So, the "brown+hare" combination just didn't break the attachment of the context to "white"... on the contrary, it made "white" closer, *as "brown" is a color too*! And now we have "color" cloud traits in the pattern, making all colors even more related - that is, more probable.

And there you go! It added "*green*". Why? Just because of the above! *All* colors are pretty much tied together as they *often* are used together. And various different colors often happen in the *same* context. So they should share some similar traits too. That's how "green" later becomes more probable than most other words - we get tied to the "color traits cloud" more and more with every new color token. 

Of course, i call it a "color traits cloud" only as a *simplification*, because in reality, it includes everything that typically comes *with* colors in text too. 

So, the model is not "stupid" to mix up or forget the color of our hare. It just *works* this way - by pulling the probable *compatible* tokens, and compatible they are primarily because of their common usage combinations, which are encoded in the traits.

Let's play more. In a European context, "white" is tightly related to "purity" (snow, wedding rituals, innocence, etc.). Let's test that too:

A hare of white embodies the -> ```moon, a symbol of purity and new beginnings.``` (seed: 1122261616)

A white hare embodies the -> ```essence of purity, innocence, and good luck. ``` (seed: 4173163411)

While making these examples i also got "springtime" in the continuations and wondered why. It turned out that the white hare is a symbol of spring in Japan :).

If we replace "hare" with "rabbit", it goes right in the direction of "Alice in Wonderland" as the "white rabbit" appears a lot in that story and the neural network saw it during training.

A white rabbit from -> ```the book Alice's Adventures in Wonderland.``` (seed: 2624672892)

But

A white hare from -> ```the Black Forest, Germany.``` (seed: 2800670882)

I'm sure you've noticed that this process is the very essence of being *biased*. Yes, as by definition it works by pulling the learned *associations* together - basically, it's a big bias machine.

What happens if we have unrelated tokens? Like, if our text is "I was swimming today. You look great". The "swimming" token pattern then isn't related *much* to "look", because these do not often happen together. And the transformer then doesn't mingle the "water" cloud much into the "look" token (or only does it a bit). So, the word "look" doesn't get *much* closer to whatever was associated with "swimming". The traits of "swimming" are not transferred to "look". 

A thing to note here, as i said before:
 - when we mingle traits, we do not mingle the complete original tokens, we mingle only their interpretations (traits) created with the V matrix that i will explain later
 - however, we do not just add or remove only the *relevant parts* of pattern interpretations (traits) that we mingle. We simply blend the interpretation patterns (traits) completely, with all the irrelevant parts they have, mixing up everything. Of course, as a result we make certain features more prominent, while some other features fade. Not to mention we change already existing subpatterns by this. We just affect *everything*, making new token closer to something new, not just to the "relevant" part of the new.

However, as each attention head learns to extract its own type of pattern traits, it learns to minimize this flaw by extracting only the more or less relevant part of the pattern for mingling.

And that is exactly what we have the V matrix for.

#### V matrix.

This matrix is the final puzzle piece of our attention head triumvirate - the Q/K/V. It learns to extract information about some specific traits the attention head specializes in. 

Of course, it is not just a copy of these traits from the donor token. It's a *different* pattern based on the input. It has different proportions, values and meaningful *figures*. We can say that if earlier in the tokens we had stars and circles, here we *may* have *ponies and elephants*. But as it actually *needs* information about the very traits it extracts, it shouldn't be just some basic signal, as is the case with Q/K.

This "figures thing" consisting of the traits is a tricky one to understand, so let me repeat everything again. 
Neural networks do not encode meaning just along a single axis of a pattern. They go an abstract level higher and encode meaning through *combinations* of these axes. And not just with specific combinations of actual values in every axis, but also in the *proportion* of the values. The proportions of the axes just make *figures*. Even more than that, it's not just figures, it's not just a pony, it's also how *skewed* it is. Which rays differ in what way. And so on. This way pattern proportions produce an additional relation on top of just similar single values.

The relatedness of "white+hare" to "snow" is not because there is some special single "snow" axis. It's *just because some of the new axis proportions in their full combination _relate_ to the "snow" pattern*. 

And it's never a single "pony" or an "elephant", but rather a huge weird figure where every spike can mean something *when paired* with some other spikes or "flats" or whatever. 

Let's call it - a zoo! Or a *meta pattern zoo* within our pattern. 

Like drawing the butterflies... that altogether form a cute face if we look at them the right way.. and a pizza if you look at them the other way! 

And the role of the V matrix here is to learn how to find and extract the traits of a concept, recorded in both value ranges and proportions of axis values (traits).
 
In simple words, it learns to see a pony in the pattern, including how much the pony is skewed. And it doesn't matter if the pony rides on a turtle or not!

Now let's get back to what actually happens.

As the V matrix extracts the traits it knows per token, from a pair of tokens we get two rows of numbers. What do we do with these? We:
 - multiply each of these numbers by the factor of "relatedness"/compatibility found by our Q/K comparisons<br>
   That is, for target token #2 and donor token #1:<br>
      &nbsp;&nbsp;V1 * comparison of Q2 and K1<br>
      &nbsp;&nbsp;V2 * comparison of Q2 and K2<br>
 - simply sum the resulting numbers of the two rows axis by axis. Yes, we just put one list of numbers over another and sum their values

And here we are - having an updated trait sum, a new figure. It has the same amount of axes, as we only summed the individual columns.

And that gives us our "mingled" pattern consisting of extracted traits from all the compatible tokens preceding the target one and the target one itself.

Basically, here our single attention head can rest, as it has done its work on every token pair. And all of the compatible traits were mingled into all the compatible tokens :).


#### A bit of a critical thinking about attention block

A dubious thing about attention is that we always draw a new pattern from the *morphed previous pattern*, instead of creating a genuinely new pattern against the background of the old one, or correcting the existing pattern. 

It means that this implementation:
 - Cannot self-reflect by definition.
 - It cannot work by contrast as it always *mingles*.
 - It cannot come up with a totally different pattern/approach, as it always inherits and morphs, but cannot change the context.
 - It cannot restart at some point producing an alternative pattern.
 - It cannot go back to some earlier point in the context and correct an incorrect mingling that poisons it by pulling in the wrong associations.
 
The existing pattern is always biasing everything. The existing context is always a part of the *new pattern*. They are mingled into a single representation. There is no memory of any separate specific isolated part of the context.


Another thing i would like you to note here is that we encode the concept of trait "relatedness" in both: ranges of absolute numbers of axis values *and* in the figures they form. 

The absolute value ranges *do* carry the relatedness concept because we mingle *scaled* patterns and the system has to learn it as another measure of relatedness. If a Q/K pair returns a *low* score, we multiply our V figure by it and make it *smaller* before mingling it into another token's figure. That means that the neural network has to react to both: 
 - The similarity between figures, expressed in proportions between different axes;
 - The actual absolute ranges of values taken by the axes - the size of the figures.

For now just note that due to scaling we also use the absolute magnitude of figures to encode information. We will talk about it more in the discussion of the normalization step. 

#### end of the critique moment


And to finish up the attention block, let's repeat again that every repeating block (consisting of attention+FFN parts) has *multiple* such Q/K/V sets working in parallel, called the attention heads. 

Each of them during training *hopefully* forms its own way of interpretation. This way, they are *supposed* to match, tie up and mingle the tokens *differently*: different (or same) tokens at different strengths, extracting and mingling the different figures from the patterns. 

And here things go weird, because *originally* each token had a single pattern/definition, but now each of the attention heads has produced its own *separate* pattern/definition per token! Each token now exists in multiple versions!

Do you think if we unite all these we will get a longer list of numbers than the token had at the start?

Not really. Yes, every attention head gets a full-width list of numbers per token. But.. they produce a way smaller list of numbers per token. That is, with fewer values (axes), a narrower table. It's a simplified representation, compared to the original input, a shorter pattern.

What do we do with these multiple simplified versions of every token?

We just take them and.. stack them together for each token! So our token table becomes much wider (than it was per head or at the start). Each row (which is one token) holds now all token interpretations from all attention heads.

And now it's as if each token were represented by its smaller reflections in multiple smaller mirrors - attention heads. This way we got a "faceted" view of each separate token. 

Of course, in reality, these "facets" have redundancy, as attention heads could grasp *similar* patterns and compare/interpret more or less the same things in the end. Making attention heads do something totally different is a separate task. Usually, developers try various tricks in an attempt to get rid of the redundancy. Like initializing the heads with different random noise, switching off some heads during training, splitting the tokens between the heads by some rules, overlapping their attention, and so on and so forth. *And here, a single flexible attention head could be really nice..* :).

But do you remember what i told you about transformers? It feels like somebody just liked copy-pasting a lot :). 

Also, earlier, i said that the context is always fixed and we can't come up with a totally new pattern. This copy-paste approach *patches* it by creating multiple interpretations where things can differ. *Yet* they all still just *morph* the existing context, the same pattern. They do not cross-talk. They move only forward, even if they do it in different ways. 

So, what do we have to do with our faceted token representation?

#### Here we finally get to the last part of the attention block: the O matrix.

Its purpose is to convert this faceted token view into *one more* different pattern, yet a mingled one. 

After all, it's the Transformers. If you want to do a single operation, just transform everything into something else! :)

> And yes, of course, each one of these transformations introduces some noise and loses some useful signal. So the more transformations we have, the harder it is to train a model. It just has to find its own similarities in each of the intermediate patterns. On the other hand, this is the actual way Transformers work, their core toolkit, allowing them to process patterns.

Conceptually, the role of the O matrix is to transform the faceted attention heads' output into a pattern that is compatible with the original pattern that we had at the input of the attention block.

Also, it tries to filter out single false attention heads' results, relying upon the results of all attention heads, thus patching the attention flaws. Remember what we talked about in the Q/K block? Here, if the heads produced some redundant information, the O matrix can learn how to use it to extract the useful signal and ignore the "lying" heads. 

And here, we finally get the required *change* to our original pattern that the attention block has produced. 

This change carries the shifts to the original token patterns, turning them into *different* clouds of traits. Clouds that now reflect *the sum* of their related traits per token. 

Phew!

#### Residual connection step

This is what we call the process of adding our changes to the *token patterns as we had them at the input of this attention block*. 

It's literally just adding their values together. 

What's the point of doing it all this way? Why couldn't we just create an updated, *ready* token in the attention block?

Well, having it this way adds a certain uniformity to the structure of the pattern. Our model has to adapt to the fact that the structure of its pattern figures should match the one it had at the input. 

By limiting the actual functionality of attention to just the gradual changes to the original pattern, we resolve the potential warping of figures within attention.

The whole attention block in the end has to work as a *fine-tuning* of the original pattern, not as something totally new or free. And that makes it much less "heavy" and much cheaper to train, as now it has to find only the *changes*.

And the mingling of the *simplified interpretations* of the V matrices still works without making everything fall apart fast. The attention result just tunes the original pattern, *rather than replacing* it.

So, in a few words, here we return to the stars and the circles we had originally, but now somehow changed, hopefully reflecting their relatedness better, as every single token now includes the related traits of *all preceding* related tokens.

And at the end of the Attention block..

#### Almost forgot... the normalization block :). 
Usually, normalization means making something less deviating ;). Not sure if this explanation helps, so let's dive into this block. It is much trickier than one might think, as it's not just scaling everything to fit a specific value range or something like that. 

This block has two parts:
 - The plain math that finds the "center/middle" of the pattern (called "mean")
 - The plain math that finds the average length of the pattern rays from that center
 - Shifting the middle of the whole pattern to the zero spot on all axes, so it's now "centered"
 - Changing pattern proportions by making its values more statistically average in comparison to each other, so we don't have any more single too big spikes or single too short ones. At the same time, we also compress the range making values smaller.
   <br><br>
 - Multiplying each axis's value by a number that the model has learned for this given axis (changing ray lengths)
 - Adding to each axis a fixed number that the model learned to apply here for this given axis (changing ray lengths)

I don't know how to comment upon this, as at this moment i feel like crying. I know this is empirically considered to be a great solution that works, but it only makes me cry more. It is all about methodology, after all.

So, let's first explain why it's used at all. 

We have several tasks here in the *current* arch:
 - to produce some stable and more predictable input for the next block
 - yet to prevent losing subtle distinctions when compressing the value range 
 (as in a fixed range, a single too big axis value can make all other values be so small that they will share the same value and lose all their distinctions, while with this method, the axes don't become equal if they were not originally).
 
The thing is that the next block (FFN) anyway sees *only* this centered and averaged representation, so it believes this is the ground truth and it learns to extract information from this data version. But it doesn't mean there was no data loss during normalization or that it's not a bottleneck. 

#### Start of the critique
Let's concentrate on the fact that we introduce two learnt parameters per axis: a value to multiply the ray's length by and a value to add to the ray's length. These are two learned values per axis, and they do the same to *any* token produced by the attention block with *any* figures inside. 

Now, let's see the problems it creates. At this point, we already break: 
 - the absolute value range (scale of values)
 - the actual figures we had in the pattern
 - the skew of these figures.
 
The actual *figures* can still be used, however, *only within a limited range of values*, as if an elephant's trunk is too long, it gets averaged and then feels like the elephant has silently visited a plastic surgery clinic to become a k-idol. Even if the model tried to encode some relatedness via trunk length during learning, it failed and had to find another way. Because if our token has no other prominent axes, it would shrink. But if our token had other high values, it would *not* shrink this much. So the value becomes dependent not just on mingled figures, but its *passage* becomes dependent on which other figures are present. And now it can't vary the length of a *single* axis, as it becomes much less predictable.

Also, as we apply the *same* gain+shift per axis to *all different mingled tokens*, the model *cannot* rely on the absolute value range from now on as this method of passing the information becomes almost unpredictable. All of a sudden, the range of an already established pattern can change once the average length of all rays has changed.

The skewing of figures cannot be used, because normalization introduces *uneven* distortion, as the border values will change more, while the average values will remain somewhat the same. It partially applies to encoding with figures as well.

All of this, as i reason here, should force the model to rely upon figures consisting of *average* similar values, and limit its methods for encoding information. 

So now, the model has to encode relatedness through proportions of specific groups of axes, rather than through all of them, because changing a single axis may change proportions and values of *other* axes. So the most reliable way to ensure the signal passes is to isolate a certain group of axes via some specific gain, so that this specific combination always passes the normalization gate with the same distortion level. The model is forced to tie certain types of figures to certain axes, as they have the specific gain+shift values it finds during training and these are *compatible*, while some other axes might be compatible with other figures.

Thus, our normalization layer is a huge noise source, significantly complicating the model's means of delivering its signal to the next block. And it's also a huge bottleneck as it limits the ways for the model to encode the meaning.

It's like telling the model: "Wait, we have screwed your patterns by compressing and averaging them, each one in a different way, so now try to find how to change each of the rays to fix it. And sorry, you can use only one combination for any of the tokens you create".

And the model is like: "!@#EY!821, Umm.. okay.. maybe i can at least try to use this to create groups of axes by giving them common boosts fitting some of my figures, then it's easier to pass through this noisy gate.. Let me then try to waste a lot of compute trying to find what passes through this pinhole at all.."

I think that all of this could be avoided if we had initially *stabilized* the signal channel the model uses.

> It could be cool to make the V matrix learn on the already mingled tokens, but it would require a lot of changes to the KV cache and it's a different story.

Let's work within the attention heads with *proportions* directly. Define a range for the proportions, mingle tokens by the *proportions* first, train the V matrix upon the proportions instead of absolute values. Whatever the V extracts, encode it then via the proportions in the same range again. 

This way we *abstract* away from the actual absolute values, tying our model to the proportions channel. And if you believe it limits the means of communication, it doesn't, as normalization makes absolute values impossible to interpret anyway. It still has to rely upon figures, it just has to find a way to *pass* them through that distortion. 

This way we do *not* need the pattern centering (the mean) and the compressing/averaging of the proportions (the standard deviation). We can just pass an already stable, uniform representation.

This way, the model can *rely* upon the proportions and their skewing, as they become a stable channel of information to the next block without all this terrible noise.

This way tokens *keep* not only their internal pattern coherence but also their *relative* coherence, as we haven't destroyed their size on a *per-token* average basis. They originally exist in the same range, and we only change the proportions of the patterns within that range.

Yes, if we implement proportions linearly, some axes can get oversaturated and clip. It may break the way training works. To avoid this, we can make *non-linear* proportions compression that will normally operate linearly over a broad, common range, but still allow going *beyond* that range too. Say, after "100" we apply a much harder compression, then after "1000" even harder, and so on, making clipping nearly impossible.

But this is only half the fun, as here we keep absolute values within the proportions stable between the blocks too! The model now can encode data not just in figures via axis proportions, but also in the actual size of the figures, as these are now stable and the FFN can use them to extract information.

I believe all this should be faster than finding the *one-size-fits-all* parameters for the normalization and finding the safe values that would pass the normalization, preserving the information.

And it's a much cleaner representation for the next block as it is minus one noise source.
#### end of the critique

And all of this is called Multi-Head Attention (MHA). 

Just one more little thing to mention. Transformers have two phases of processing data: one is the "prompt evaluation", where the existing text is converted into an internal representation, and the other is inference, where the new tokens are generated. 
 
So, if during prompt evaluation the target of traits amalgamation is every given token, we just enrich every token with traits from all related preceding tokens. During inference, we continue that by making the new, non-existent token a target for the sum of the whole context.

And upon this, i think we have finished the Attention block and can finally move to...

### The Feed-Forward Network (FFN) part.

You may have heard that the FFN block stores *knowledge*, but... all of this is not really accurate... to be precise, it's not true at all :). 

To understand the role of the FFN, you need to know only one thing. At the input, the FFN receives the token table from the multi-head attention block... and at the output, it produces a pattern that is added back to the very same token table it got at the input. Which means... its output should be compatible with its input :).

So, as you can see, the FFN just once again "updates" the existing token patterns by doing something on the inside. 

Let's see what it does.

The FFN is composed of three parts:
 - The first matrix - data extraction.
 - The activation function (passing gate) and bias.
 - The second matrix - converting the data back.

First, let's take a look at what happens:
1. The resulting token pattern from the MHA is multiplied with the first FFN matrix, which has way more pattern axes.
2. A unique static numeric value (bias) is added to the result (per axis).
3. The passing gate (ReLU or something else) passes through only the *new* axes, which are actually related to the input.
4. Filtered new axes are multiplied into the second matrix, translating them into a different representation pattern with the original number of axes (as on input to the FFN).
5. A second static value (second bias) is added to the result (per axis).
6. The resulting pattern is then mingled with the original token table coming from the MHA. It's a residual connection, just like after the Attention block. So the final result is actually a mix of the MHA view with the tuning from the FFN.

So, conceptually, this operation finds the relevant figures/patterns stored in the FFN and then converts them back into the figures compatible with what we got from the MHA, and mingles them in. Just like earlier we were mingling different tokens, here we mingle our tokens with FFN figures reflecting.. what? Token clouds? 

Nope. FFN does not really store some separate standard token clouds. This is because it has to produce only an *adjustment* to the input pattern, not a *new* pattern. Remember, we mingle its result back into the original input, so it should learn to *adjust*, not to *translate*.

So, what the FFN stores are patterns on *how to change* a typical input to shift it closer to some standard pattern. In other words, it finds the most probable *adjustments* required for infusion into the MHA pattern, to make it closer to some common token clouds. The FFN learns to adjust token clouds - to brush them up - but it does not learn a separate language of "standard" token clouds.

And if you wonder why we need the O matrix earlier, or why we couldn't just feed the MHA results to the FFN directly, this is precisely the answer. The FFN needs to have *the original token representation with the MHA changes* to brush it up. It can't work off the pure MHA changes, as they don't have information about the actual related token clouds. They have information only about changes to these clouds. And the meaning of these depends on the *original* token figures. In order to mingle the MHA output into the original token, we first have to translate that faceted representation into a single one. That's why we have the O matrix back there (apart from noise cancellation).

#### How do we mingle the MHA into the first matrix 
       ..or "i've finally decided to explain what matrix multiplication is" :). 
       
 - Every operation is performed per-token, so we do all these things per single token of the context - per row from the MHA;
 - Each token coming from the MHA is a row of, say, 512 columns/axes;
 - The first FFN matrix in multiplication has just as many columns/axes/dimensions as the MHA row has - 512 in our example;
 - However, it has way more rows, like four times more: 2048;
 - We do *not* multiply the FFN matrix "into" the MHA row. We multiply the MHA row *into* each row of the FFN matrix, by multiplying every column of the MHA into the same respective column of the FFN matrix. And we do it with *every row* of the FFN. So the same one input token row multiplies *each* row of the FFN matrix;
 - Then we sum up the numbers of each of the 2048 FFN rows and get 2048 new values/axes/dimensions;
 - So we have turned 512 axes into 2048 axes and got a new, four times larger pattern;
 - We do it for each token we have, returned by the MHA;

Basically, we did 2048 dot-product comparisons, finding how similar each of our tokens is to each of these 2048 patterns.

> This is usually called "matmul" in slang or "matrix multiplication" and that's what we mostly use as a method to extract some information from a pattern, or to "translate" a pattern, which, in a way, is the same thing. A lot of people also say "project into another space", which is even more confusing, as they don't even mention if they use an LCD projector or a DLP one. humor.

But what does that process *actually* mean? Well, it's pretty obvious. 

To understand it, we just have to remember what our MHA-returned tokens do represent. They represent the pattern configuration used for expressing their relatedness to other tokens.

When we mingle these into the FFN matrix, we simply complete each of the 2048 FFN's unfinished patterns, each serving as a "test" for:
 - "how well"
 - this token matches
 - a specific trait part
 - of the *most probable adjustments*
 - to the traits of tokens' relatedness,
 - that the FFN's first matrix has learned during training.
 
How does it find these required adjustments?
You can think of these as a lock-and-key mechanism or like a fun test in a magazine where you answer the questions.

The MHA-produced relatedness pattern just "unlocks" some of the FFN's "locks", by being compared to each of the new pattern's axes (the rows of the FFN's first matrix, which become axes of the new pattern) - these books/locks. It's just the way we compared shapes in Q/K - a dot-product comparison; here we just do it against 2048 different test patterns/rows. And voila, each new axis/book gives us a value showing how much our MHA-produced pattern "relates" to it. 

The incoming token here works more as an original noisy signal coming through a set of 2048 *detectors*, and based on that, we get a new profile of that signal. It's not even quite amplification or dampening, because in the end we get *different* axes/figures. It's not a simple "upscaling" of the same pattern. 

It's like getting the results of a psychological test with 512 parameters and mapping them onto a totally different psychological test with 2048 different parameters, finding which ones match and to what extent. 

Our results here are the "scores" of that new psychological test, in a totally different parameter representation. This is the pattern that interprets the data in a different way, extracting the information required to produce the necessary changes to the original pattern.

As it holds the refinement to the original "profile", by injecting it later, it can even shift the token "profile" to a different "final diagnosis" :). That is, to amend the original traits in a way that moves them from one cloud of traits to another. This means not just making the existing token have a more identifiable shape, but changing its shape to a different "proper" one. Although you have to remember that the FFN operates *per token* and has only the context embedded into the token by the MHA, which limits this functionality - but more on that later.

#### But... how does it decide if the new axis is related to the original pattern?
In a classic implementation, the model just learns a simple fixed value that signifies the edge to cross to be considered relevant. If the result reaches this edge value, then it means the "Q/K" of the FFN says "yes" - this new *axis* is relevant, let's use it in the new pattern. If it's not, we just do not use this axis from the FFN at all and output a zero.

This fixed value (which can be negative or positive) is used to detect the edge and is called the "bias". Of course, it's just a number learned during training. 

Basically, we just add it (e.g. -2.5 or +4.1, etc.) to an FFN axis value and it just "negates" the typical noise value of the learned pattern on that axis. 

Then we can check if the sum is now more than zero or not. If the result here surpasses that noise volume level, it means our input data had its say in this. If it's less than or equal to zero, we believe all we have is just the standard pattern-level "noise" we got after multiplication and this trait is unrelated.

Please don't get me wrong, we do not compare the result to the bias value itself. We *add* the bias value to the result and then check the sum of this operation with a fixed non-learned function (ReLU, etc). In the original implementation, it just checks if it's more than zero or not, as i explained above.

But conceptually, that's just what bias does: it just serves to negate the standard pattern-noise level, so we can know if our own data is relevant to that axis.

A funny thing here is that the bias may be so high that it can force the enrichment of *any* input pattern with a shift typical of a specific token cloud, that is, towards a specific "idea".

Or it can be so low that it will make shifting towards certain "ideas" nearly impossible, unless it's a very rare case.

But in practice, as the patterns rely on multiple axes, this is not very likely. And of course, these factors also get balanced during training.

It also mixes two concepts into a single fixed learned value: 

 a) deciding axis relatedness by negating the value - a "one-size-fits-all tokens" approach.
 
 b) signal amplification/dampening that distorts the pattern as it is. 

But as we do not directly infuse that result into the original input, and it's a fixed change that is always applied, the second matrix learns to adjust, so that distortion shouldn't go further. 

So, we have added the bias value to the comparison value per axis, now what? How do we tell if the axis is related? If the lock is unlocked? How to know if there is any relevance between the MHA-learned cloud of tokens *and the traits* that mark FFN-learned most probable *adjustments* to shift the pattern closer to certain token clouds? *sigh*

> ...the irony of this architecture is that to measure if there is some useful trait/signal in the incoming token, we first *multiply it* and only *then* check whether we had to do the expensive multiplication at all :)

And that's exactly what our second block does - ReLU. It tests each of the new 2048 axes with the added bias to see if the value they have now is positive. If it's more than zero, it considers this axis/trait as a related one and keeps its value on this axis. If the result is negative or zero, it *nullifies* the axis value - our token is not related to the trait of this FFN-learned adjustment. In this case, the comparison result is just discarded in this axis.

Once we pass the ReLU filter, we come to the second FFN matrix.

#### Translating it back into the same pattern language
And here we do a "reverse" operation: we multiply our 2048 axes of a new pattern into the second FFN's matrix. That one has reverse shape: 2048 columns and 512 rows - so it converts our adjustment pattern into fewer axes, into a *different* representation that is compatible with the input pattern language, so we could mix these.

If:
 - we had only *one* repeating block of attention+ffn
 - used the original Input block with tokens for both input and output matching,
then here model would learn to translate these 2048 parameters back into the representation compatible with what we had in the very beginning. Axes here would mean the same thing that they were at the very start in tokens vocabulary, the figures would be similar - stars and circles again.

##### hopeful dreaming
However, as we have *many* repeating blocks chained together, model has no need to make this internal representation uniform. It has freedom to find very different token relatedness traits with every of its block. But the closer it's to the end, the closer it has to be to the original vocabulary pattern language.

So if on start it marks tokens by stars and circles, in the middle of repeating blocks it may mark tokens with .. circles and stars! Why not? It can choose anything :). Worths paying for the extra training!

##### sore reality
Well, the truth is.. it *could*.

In reality the creators decided that too much freedom is not practical and.. added results of the original MHA output to the output of FFN block, as said in the beginning. 

Thus:
 - enforced a somewhat uniform format between these blocks, as data this way has to keep more or less similar representation, otherwise addition of the input data would break the output. They have to stay compatible.
 - FFN is freed from the need to keep original signal in its representation, it can concentrate solely on the tuning and return only the *corrections* to the original signal. Actually, this way FFN is enforced to do mostly finetuning of the existing pattern, because otherwise addition of these two would break things.

And they called it "residual connection". Why not BLEEOR?..<br>
  .."Be like everyone else, or else.."

Getting back to the actual process, we have passed second matrix and.. we add the second bias value. It's just a numeric value the model has learnt per axis. I don't really see any conceptual meaning in this operation, apart from pure speed up in training, where model can just quickly adjust the typical result it gets in some axis without changing the whole similarity representation. Of course it's a crutch in a way, but it works. 

So, once we have summed up tokens data from before FFN and after, we again normalize the values. And here you can just reread the normalization after the MHA as it's the same operation and the same critique applies here. 

Finally, we send this result to the *next* repeating block, which starts with its own attention block doing all the same again. And again, and again until the last repeating block is done.

When it's the *last* repeating block, the result has to be compatible with the original Input block vocabulary. 
> unless the model has a separate output vocabulary layer, then it has to be compatible with that one instead

In a way, whole FFN is like a kick for the tokens to get closer to some specific clouds after the attention mingled in all the previous tokens. In my opinion it's mostly an error correction mechanism to fix results of MHA amalgamation. No jokes, it's a hard task as it needs to translate the chaos of infused traits back into recognizable and relevant shape where certain figures would get the clear priority. And the key thing is that it can actually learn (and should) the selection of the most relevant cloud of traits. But that's about it. Imho, most of the actual information is stored in the tokens patterns. And in the way MHA mingles them. Here, imho, we just brush it up, despite the huge size of the block.

There are experiments where people edit weights in FFN and make it produce different desired output, like changing the name of a city to some question and so on. Yep, why not. They merely change the detection and adjustment of some token clouds traits. The very city name comes from the token vocabulary. That token of the city name has a pattern with its traits. MHA produced something vague. FFN refined the traits properly to align it with that specific city. Engineers managed to find how to maim the refining mechanism to shift its adjustment to a different traits cloud, a wrong one. City has changed. Does it mean FFN stores facts? I don't think so. How could it even do that, considering that it works *per token* so it doesn't have *any* context for the right fact.. all it has is just a mingled pattern.. sigh.

You could say that the token already has all the context mingled in by MHA, so most of the context is in the single token! But the thing is, it's true only partially. And besides, the mingled in traits are not *ordered*. It's not a sequence of words that makes sense, it's just a cloud of traits where model needs to filter away the noise and refine the important traits, maybe finishing these up or dampening the other ones.. And it never means a single word, it can be tied to lots of unrelated things conceptually.

If you believe it stores *facts*, it means it should know the actual patterns for the *facts*. Remove the residual connection then and see what happens.. Will the patterns produced by FFN replace the original token reliably? Or, after all, it's a *function* to patch the data, not a *store*? 

Getting back to the conceptual level, as you see, we do *not* store anywhere the information or "knowledge*, in sense of facts, symbols, sentences or whatever. We simply learn the required adjustments to the pattern representing relatedness to clouds of tokens. We did it in attention, and now we did it in FFN. Is it redundant? Yep, it is. First, the attention adjusted tokens, and now FFN adjusted the adjusted tokens.. And that's exactly why we can always *distill* a trained model, to get a way more compact form that performs close enough to the original. 

 
### Critical thinking. 
A thing about FFN is that by adding residual connection we enforce the *same* token clouds to be used in intermediate repeating blocks. Model can not develop a different abstract data representation in the middle, because it always has to be largely compatible with the original input format that already has its statistical distribution language. 

And we drag it across the whole repeating blocks. What's the result? We should have way more repeating layers to give model a chance to develop critical changes to existing token clouds by introducing these in a step by step fashion through the redundant blocks. 

And then it has to do a *reverse* to come back to the original pattern language so we could match the pattern against standard vocabulary with its token clouds. 

With residual connection we cut off a chance of model to efficiently rediscover alternative token clouds, forming *stable* new representation holding across several repeating blocks. Every repeating block can step just that much from the input data as it can only refine the existing original pattern in the end. 

That means FFN doesn't really do a parallel abstract understanding, it *tries* to make it emergent but it left with almost no chances as it's engineered to *refine* not to *restructure*.

Could it develop a different intermediate language be there no residual connection after the FFN? Probably, yes. But it would certainly require way more resources to train and probably multi ffn blocks to avoid fast degradation. 

And then we probably still need the residual connection at some point deep in the repeating blocks, to keep its own parallel interpretation stabilized. 

On the other hand, in the current implementation every next repeating block's attention grasps different relatedness clouds. In a way, it *is* a parallel abstractioning happening sequencially. 

The thing is, that next repeating blocks can operate upon compound token clouds assembled in the previous repeating blocks. This way next layers can step up in their abstraction level, already processing not just "went" but "red haired person went" and to tie traits happening only within of this context, related to this cloud of traits. However, its prediction is still limited to the original tokens sequence distribution statistics and somewhat tied to the original vocabulary. Also, the size of matrices in the next blocks is the same as in first ones, but the complexity of traits to capture grows, which should be a bottleneck for developing really complex abstraction systems different from the first layers.

##### end of the critical moment

### Output block
When all of the repeating blocks did their work on the pattern, we finally compare it to the vocabulary patterns (output or input layer), to locate the token that resembles our new pattern the most. And that would be the closest set of characters representing our newly produced "idea" frankenstein :).

This process is a bit tricker than input as there can be multiple candidates with a similar patterns. For example it can be "ten", "10", "*ten*", "Ten", "_10_" or even.. "9" and "11" :). It can be "Hello", "Hi", "Hey!", "Heya", "Greetings", "What?", "Leave", etc. As explained above it can even be very different things, totally unrelated by human ideas, like: The/My/Every/It/Black/Always/etc. 

The comparison score of the new pattern to vocabulary tokens is called "logit". 

And here goes the saddest thing of transformers: samplers. What they do is decide which one of resembling patterns to choose as the actual token. It decides which characters we will see..

But how? 

#### Samplers
Samplers are a very ironic thing within transformers, as they do a very important job of a final word choice and yet have zero AI, ML, DL or any other neural network related stuff :). It's 100% plain good old conditional algorithms having zero idea of what's going on, zero learning.

Samplers merely take the logit scores for all found compatible tokens and then choose the one according to whatever samplers user has chosen to apply to the generation process. These samplers are "dry", "top_k", "typ_p", "top_p", "min_p", "xtc", "temperature" and so on, they are often updated, some new stuff is added.

The exact order of samplers, their lists and implementations differ across various engines, as i said it's *engine's* thing, not the neural network's part. So i won't go into explaining how exactly they work, just the idea. 

They just do various stuff like normalize the logits with softmax, flatten the scores, filtering out tokens that were used too much earlier, remove tokens that have too low scores to be statistically trustworthy and so on and so forth. And then they choose one from the remaining ones.. by flipping a coin :). I believe it's the most ironic thing in the whole transformers implementation :). We first put in a hell of compute powers to detect the most fitting next word and then we just randomly choose one from the candidates :).


Once it's done, we just inject our new token/word into the first step and repeat the process, going in our quest for the next syllable, word, number, comma, space, dots ..

When does it all stop? 
Well, sooner or later neural network produces the *ending token* and the engine running it learns to interpret it as a stop signal. How is this ending token happening? The same way, it just becomes probable as it is something model has learnt to do during the training, it was in the datasets. Usually it's something like "EOS", e.g. <|EOS|> which stands for "end of stream". 

Well, this one was short :). 

So, let's now come up with some afterwords..

### Conclusion on transformers
I think hardly anybody would expect logic to emerge from all this process, which in fact is a simple abstract traits mingling based on resemblance, which in its turn is based on statistical distribution of data chunks that are not even meaningful in our sense of it :). 

To put it in a couple of words, what we have here is called "associative thinking". The only difference from humans is that it associates *syllables* more than words.. sigh.

It just adds up the associations of syllables and words that the model has learnt during its training and that's all to it.

Same as if somebody asked you to tell the first word that comes to your mind when you hear something. 

Yes, deeper repeating blocks process compound constructions, that is already associated syllables that form something that is much closer to human concepts. If you remember i said in the text that this is how it overcomes the tokenization issue. It goes more abstract this way and operates on meta-concepts of assembled tokens. But the key limitation is that it still is limited by the conceptualization / traits clouds based on tokens sequences. And whatever deep abstract level it develops, it just has to use the clouds that reflect probable character sequences. It has to stay close to these as we just mingle in results of every next layer and in the end we expect these to be still compatible with our only truth - original clouds of tokens reflecting characters distributions.

It's like trying to make model think in quotations of popular verses, yet to make sense of the world.

Can you be logical thinking this way? Obviously, only through a very, very tedious brute force training and when you do a step away it all falls apart. Which is exactly what we have with transformers.

#### A fun proof
Now let's do a fun experiment. Let's send to llm only 2 tokens: "The\n\n" - simplest pattern. Nothing else, **no template**, and at that let's use instruct model. 

I started with Llama 3.1 storm 8b (it was just the first one in my list) and guess what? 

The candidate #5 for continuation is ```The``` :). This is what model believes should go next. Why? Because the only changes we had were from V matrices interpretations and FFN matrices refinement, so within candidates we have gotten the same output that was originally in the input, plus the bias skew and associated meanings. But the original pattern was so little distorted that our continuation suggests the same pattern - "The" as the 5th most probable option :).

Now, guess what happens if we send "The\n\nThe\n\n"? :) Yes, it just continues the pattern, producing endless ```The\n\n``` as the first candidates, reproducing the same pattern, strenghtening it with every next repetition in created resonance.

It just copied the traits from the original tokens and went into resonance, traits of the same token became more and more prominent, making everything else improbable. It's just an echo of itself.

Next i tried Gemma3 4b and.. guess what? Its top candidate is ```The``` right away :). So, the reply i got is just endless loop of "The" :). Llama could extract from "The" some other probable token cloud traits, while Gemma implementation clearly aligned with the strongest traits and in the end, ironically, it worked even worse for this scenario :). And of course with every next "The" its probability grows a lot, leaving all other candidates less and less probable :). It just falls into a pattern of dotted line, where "The" is just a dot it uses.

For Gpt-Oss-120B "The" is at #4 but again if you prompt it with "The\n\nThe\n\n" it just loops with endless ```The``` :).

For Qwen3-Vl-30B-A3B-Instruct single "The\n\n" is already enough to make it #1, so it's just a loop of ```The\n\n```.

What would you do if somebody was stuttering while trying to say something? Would you reply with the same word over and over? :) Not if you are adequate. And that's exactly, what transformers lack - adequacy. *More on that in the next chapter :).*

So why don't they do it when we use instruct models *with proper templates*? Because the template part establishes a different context for the model, enriching the input with its.. personality :). All this template formatting we add to the prompt is not only establishing a chat pattern model knows - question/answer, but also embeds into next token its favorite "ideas"/combinations captured within the traits of these template tokens.

#### Final words
Transformers are not about logic, it's just *incremental associative morphers*, tweaking your input in a most probable way through associative thinking, that it learns from the training data's internal structure :). They merely continue your pattern the way they learnt as most probable, without being able to *reflect* upon own  decisions, to *go back* or to fully rely upon our actual *human* ideas.

It's hard to be surprised when something provides no logic or hallucinates, when all it does is just uncontrolled associative self-morphing of a mix made of both syllables and words :).

And i'm sure after reading it all you should be full of ideas on how to make transformers a much better thing :).

#### An afterword
Here i'm sure most of you want to say: "Hey!! What?! It's over? Why didn't you tell about the most obvious flaw of the transformers?! It's clear as day and you ignore it!"

Well, please forgive me, it's just that though i'm sure it's clear for you, but it wasn't as obvious for me and took some time to blossom into this small addition.

Yes, you are absolutely right, the main issue is that token patterns mix two different things into a single represenation:
 - characters sequence
 - conceptual similarity

Let me explain.
When we process tokens, the only goal for this is to transform the *existing* token pattern into the *next* token pattern. However.. we find final matching pattern by its shape, that is, similar tokens should be interchangeable. In other words tokens with similar patterns should *all* fit the context. And it means that they all should be semantically close to each other, right? You can't replace a word "red" with a word "tea" without breaking the meaning. 

So the token patterns have to develop similarity based on the *meaning*. 

*Yet*, the very essence of transformers is to tie tokens based on the *probability* of their usage *in a sequence*. Or, in other words - which token *usually* goes next. And that can be *irrelevant* to the actual *meaning* of the sentence.

See? The very same token pattern should encode two very different things: probable tokens sequence AND "synonyms" at the same time. So the same pattern has to encompass both: "comma" and "crimson", for example. As they both are *tied* but one *as a synonym* and another one *as a common next token*.

These are *totally* different roles. And that is exactly why LLMs struggle to develop conceptual understanding. Next token *can not* be used *conceptually* if it's not *probable* just as a common letters combination. And a conceptually similar token can get rendered as something *totally different* just because it often happens in the text together and has similar traits.

This leaves a *very* narrow slit for models to make sense, they can't use *any* word fitting conceptually, they have to use a word fitting conceptually *and* being commonly used in the sequence of letters. 

They struggle to make the same tokens pattern language to encode two absolutely different things: conceptual understanding AND rendering it as a sequence of text. Patterns similarity is working by two different axes simultaneously: proper sequence through similar patterns AND similar concepts through the same similar patterns.

And it has *no* way to know if it continues the text conceptually or just statistically. The first case is always a flash emerging over the basic statistical implementation.

And what a flaw it is :). 

I'm sure, of course, you were understanding this thing without me from the beginning as something obvious, but i only had a vague understanding that something is mixed up here when i was starting to write this. And only in the end i finally managed to induce this :).

### Chapter 3: or let's hallucinate together

This chapter is going to have a jumpy structure as i just want to share some of the ideas coming to my mind when i think about transformers. As i have no way to check these, they are rather theoretical :). Also, i ignore the  compute cost that these might introduce. 

So, don't take these as "ready proven right solutions" or as "this is the only final right way to do it", these are just vectors i see and some raw initial suggestion of doing it.

Also, if you just scrolled here, this is not a complete list and several big ideas i embedded into the previous chapters.

Let's start:

#### 0. Animal consciousness. 
What makes humans be a different kind of species from all the animals on our planet? Hair style? Shoes?

It's the language. The second signal system we have.

if baby is lost in a forest and grows up with animals, such person perceives the world as an animal, not the way we do. Socialising of such people is not an easy task and often not even fully possible. 

Why? Simply because our neural network trains *with* the language, not just with experience. 

Language defines what objects/phenomenas we see as actors, what *interactions* between these we see and more importantly which ones we *do not* see.

If your language has 40 different words for different snow states, you can see and understand about snow much more than if all you see is just "snow". If your language sees no difference between some colors, these are "same" colors to you when you refer to them. A psychologists sees a lot of things happening in the mind, while some person doesn't even know what's going on inside. Think about words for logical operations, for personality traits, for people, for physics and so on. What is it we do not see?

Animals can express fury, danger and so on with sounds, these sounds carry ideas but they lack *specific* information. 

Animals can't say "there is a yummy brown mushroom under the bush if you go 2 kilometers to the east". 

They can only express pleasure and enthusiasm and call you to go there. They can tell you it's danger from the sky or danger from the ground, it seems to be specific but you can not use the same sounds to express a different state.

While language provides us with specific information beyond just *states*.

We can say in a happy voice absolutely terrible things and to say in a terrible voice the best thing ever. Like when boss tells you about raised salary. You wouldn't guess it by the tone.

First signal system, unlike the language, can express only a vector changing the state of our consciousness, giving it certain state. 

Does it remind you something? Yep, transformers. And *music*.

As humans we can do it too but..

We also can actually operate on a totally different level: through the language we can refine our thinking even *without understanding* of what we do and to do.

Children can solve equations in school following the rules yet having no idea how it works. Yet getting the *right* result. Just like LLMs..

But, we can actually think about abstract matters that never happened in our life, things we have no experience for, and yet to understand what we are talking about. Even if it's *purely* abstract. 

We can discuss concepts of perception, love, climate in another country, alien civilizations, space travelling, quantum physics, knitting and so on. 

We could never experience it but we can imagine it, share with others, split apart and make a working model of it due to *language constructions*. 

We can manipulate by the abstractions, ideas, take apart concepts and make new ones, find logical gaps and synthesize probable solutions while we use our language. All with thinking!

How to achieve it with LLMs? Through ideas distillation. 

After the training, llm knows "white", "wet", "human" and so on. However many of the words are not separate tokens but a *mix* of sub-tokens, of their traits combination. In result, llm manupulates not our words but often by the syllables. It is deprived of the access to an actual language, we literary convert our language into a set of senseless chunks to confuse the model. And as if it's not enough, we make it believe that "comma" and "shoe" are interchangeable.

How to fix it?<br>
Well, it's more compute.

##### Attention level 1.
Is what our current transformers are. It tries to develop abstract ideas but they are always tied to the actual character tokens, even deep in the repeating blocks when they are abstract, they manipulate traits clouds developed with character sequences.

##### Attention level 2.
Once we have a ready llm we can just freeze it, and add more matrices for developing the *next* layer of attention - words level. This one should contain only the separate words, not syllables. 

We have two ways to create it.
The easiest one is just to use stacked sub-tokens making the words that the second attention layer would see as separate own tokens. But it would take loads of resources.

A better way would be to add on top of it a morphological tokenizer that would encode the base form of the word and its gender/case/tense/plurality/emotional suffixes/etc as a single axes in the pattern. This would reduce the vocabulary a lot. Why can't we just use this tokenizer at the first level of attention then? Because people often mistype and use non-existing words. There should be a first layer that works for any combination of characters, tying these to specific patterns model has learnt.

Of course we would need a translation layer to be able to match tokens of this layer to the first layer's stacked "full words" back and forth. 

This second attention level doesn't really solve anything. We are still tied to the specific word sequences. However training upon this level already is split per *human* concept. And this means we will have token clouds expressing relatedness of human ideas, not of syllables if we do inference on this level.

Partially we already have it in deep levels of the repeating blocks. Partially, because we are still tied to the character based tokens there and because our attention implementation is scattered all over the tokens, it doesn't unite concepts per-word, it does something totally different it had learnt during the training.

When we have the second level, at the inference we first use the first attention to convert into patterns, translate the evaluated prompt into the second attention level and then we re-evaluate the prompt on the second attention lavel where the conceptual units are words already. 

Once we've done the inference at this second level, we get a generated word token. We translate it down and do the inference on the first level while attending to that translated token. We can do it for example by first sampling the list of candidates matching our second level token and then inference of the first level would only choose the best matching token among the candidates. We need it to keep the style of the conversation we have, but i guess here it's not really necessary.

So far we have not achieved much, we just have spent loads of compute to build a system that is much closer to human way of conceptualizing the data, yet it will probably sound nearly the same as before, as it's still bound to the sequential distribution of the words.

What should we do? Of course to jump to the semantic level, conceptual level. The one that would consist of human *ideas*, not of specific words. 

How to do it? 

##### Attention level 3.

We just need to have a "distillation" of conceptually similar *words*. Our *united* conceptual tokens.

Our languages have a lot of concepts that are expressed in various ways, yet conceptually they represent the same. They are just nuanced in various ways.

English is a poorly morphed language but we can look at: go/walk/stride/drive/fly/move/crawl/etc as an example. 

These are different in nuances but conceptually they all express motion. And you can think of the synonims dictionary that encompasses loads of similar concepts. 

Of course we should not omit important things like gender, case and so on here, but we should use these as just small flags, the way we did at the second level. Many languages conceptualize things very differently basing on tenses, gender and i don't even mention tense and emotions, these may change the concept totally.

Any LLM already has a lot of these concepts developed, the only difference is that it can't separate these from the statistical distribution of sub-tokens and in result inference depends not just on these concepts but also upon *which* sub-tokens were used to express them. 

The goal for training would be to match all the conceptually similar words but not something else. For that we just need to build a third level vocabulary of conceptual "synonyms". This task can be automated with an llm but it's more tricky than it might seem. A hack here can be that we could use the existing inference and to look up synonyms from the model's existing logprobs, filtering them with another llm. That would keep compatibility high and speed up the training.

And of course we would need to train a decoder to translate these conceptual tokens into matching the second level tokens.

And here we already get qualitative difference, as we are not tied to the *characters* anymore. We have a level where the model can produce sequences regardless of the actual words or characters used. 

Once the prompt evaluation is over, having filled the first, then second and then third level, inference starts *from* the third level and goes there, while the levels below work merely as renderers of the conceptual representation into a specific language and talking style. 

Then, first attention will capture the probable statistical distribution of the syllables. But third learns patterns between the *semantic ideas*. And it means that it won't be tied to actual *"words" as letters* but it will find concepts uniting words, that is their semantic unity, even if they are written differently. And that is already *language* thinking. Yes, to think in a language we have to go one layer *above* the language, as words are just vectors for concepts, not the concepts.

You could notice that here we do a first step away from a specific language into the abstract realm. But we are still tied to the typical *ideas* order in a language, like Subject-verb-object and so on.

So, we have our three layers and now the model can actually start *making sense*.

> You might say that all this job is made by attention+FFN, but it's not. It might be a flash of emergent feature but not a stable thing. Again, it doesn't mean attention+ffn can't develop this, it can to a certain level, it's just very compute demanding and still very *noisy* in result. Conceptual content gets spread as a thin feature across all layers trying to survive upon the basic logic relying upon syllables distributions, which may not align well with conceptual distributions.

Do you think that's it? Sorry, it's just a start :).

The issue here is that we still are in the realm of single concepts. And to make sense, you need to be able to go above and to *unite* concepts into meta concepts. 

And then we can finally untie from the language constraints of ideas ordering.

And that's why we have to go one level higher and to create the fourth level of attention.

##### Attention level 4.

One that would give us a way to distill *conceptual* structures, not just concepts. Once we have a vocabulary of *concepts* we can finally start uniting *ideas* into complex concepts. That is, we already "distill" sentences and paragraphs of data to build up typical compound concepts. 

This thing will grasp the whole *idea* of a thought we are saying with a sentence, instead of the ideas inhabiting the sentence. 

I call that third layer "gestalt" layer, as it's a single pattern encompassing the whole picture.

This is how our own thinking works. We do not think in words usually, we first grasp an idea of what would work here: 
 - "i need to greet them"
 - then we have the list of ideas implementing it - I am, glad, to see, you.
 - only then we find the right words, when we already *know* what we are going to say - including personal details
 - and then we finally can pronounce the words as characters in our own fitting style

This system is fully untied from the character sequences dependency and can truly think in human concepts and to develop typical patterns uniting human abstract ideas. And it's also untied from the language grammar restrictions.

As an optimization we do not have to follow *all* layers once we have the system developed. We can skip the 4th level or we can try to skip the second level and so on. But that's already when all of these are trained one by one.

I dunno, maybe it seems naive to you and yes it requires more training and compute, But..

This thing is *so obvious*, that when i just don't understand why it's not used. Maybe i just don't understand something, but when i see companies throwing billions into training of just one more sequential single attention transformer in hope it will somehow magically overcome the bottlenecks and just add MOAR, MOAR parameters lineary, i just feel like: sigh. People, come on..

#### 1. Meta patterns or self reflection and logic.
One more very important thingie. Current models are linear in sense of information creation, 
which is obviously no good. And an obvious conclusion is: there should be meta attention.

Let's take a look at it from another side. What we need to provide LLMs with, is to give them a way to *structure their own attention*. I call it - meta attention, which should be *trained* upon the normal attention of a model. Yes, it's that simple. 

The profit is obvious - you give LLM a chance to create patterns for patterns. That is, to create probabilities map for their own developed associations. 

And that's what the logic is. It's a way to origanize our abstractions into a system.

Of course, it's better to do it with conceptual tokens but it should work with regular ones too, improving it a lot.

All that is necessary is simply "attaching" one more llm with its input as a completed attention space, with editing rights, say, per sentence.

This is so simple conceptually, that i don't know what else to write here. Just a way for llms to create patterns for llm patterns. But of course in terms of implementation it would require lots of experiments to check this up and make it work.

Of course this training should be a second stage, once model develops associative capabilities.

I also believe that it could be very interesting to try *diffusion* models for this task, as i think they should fit better.

My guess is that this might allow llms to develop the logical capabilities if it's paired with extra training of the second llm while making the first one to solve logical tasks. It will learn to correct the structure within of a ready pattern, instead of building a pattern. A totally different functon.

This idea can be paired in various ways with the previous one.

#### 2. "Reasoning" models short overview.
Let me explain why "reasoning" models are not really "thinking" in the sense of analysis.

Main "mistake" of the "thinking" implementation is that it has no way of going back, rewriting its own situation modelling.

As the result, attention is poisoned with noise and has to struggle against it by adding more and more information, until it reaches the point where signal is strong enough to overcome the noise.

In other words, what reasoning models actually do, is they expand the original prompt by trying to fix it by turning it into most probable form they already know, by adding various most probable traits to it. 

So that in the end they could produce the most probable reply. They just try to expand or amplify the original signal, to react the way they can. 

However, what thinking should really do, is to detect the actual info plane, detect actors and interconnections and then predict the reply.

It's a *totally* different process that requires separate attention pools, choosing a proper abstractions plane and then detecting where this system goes to, adjusting its understanding until the model feels confidence or trying an another one from scratch.

For example, if the request is:
"Guess what will i do when my nose is blue and i have one more bottle?"

It may be: medical plane, physical actions plane, humor plane, drama story plane, etc. 
 - first it needs to detect a proper plane
 - then, when it puts itself into a proper context, it can start the situation modeling: what will happen next?
 - then it can distill and rewrite the pool until it compresses the interpretation throwing away all noise
 - then it can try other attention pools with other biased context attempts - planes
 - then it should just mingle most relevant results of each plane

#### 3. Context zooming.
We could implement a mechanism to dynamically manage the context. That is, we can just compress a whole paragraph of text where people meet to a "greeting" pattern enriched with some nuances of all tokens from the paragraph. 

Then, if this compressed token is highly related to the next one, we can *expand* it for calculating of *this* next token and then to compress it back for the next tokens. This way we can *zoom* into context when needed, otherwise compressing it a lot. And relatedness is just detected through the q/k as usually, as if we probe the last token of a paragraph that holds all the pattern traits of its content. 

We even could test its relatedness upon compression and enrich the pattern with specific traits for better relatedness detection. 

Of course there are limits on how much we can compress to keep it recognizable and of course we will meet some loss here, but the overall mechanism i see as quite sound and saving a lot of compute during the inference. It won't save memory as it's slow to really change the kv cache, but we can just use it as additional layer for kv cache. In the end, it's a question of balancing the load.

### Postface
If you have made it here, i would like to thank you for reading it :).

To be honest, most of it i wrote in one day and.. and then i've spent 3+ whole weeks just editing it, fixing my mistakes, adding more and more details and examples. And it really wasn't easy to motivate myself, you know :).

I know i could make more and more edits making it be a much easier text to read, adding more info.. but.. i don't have infinite time to work on it :).

As i wrote all of it "from the head" (obvious due to my poor english), it was *zero* "vibe coded", zero "copied" from other sources.. so i'm sure i could do mistakes. In that case: well, i'm sorry :). Nobody is perfect.

To check me up i used Gemini (Google AI search mode). It was also great at disputing and it caught some mistakes i made :). 

..even tho sometimes it was just confusing me and i had to be patient then with my own explanations.. :)

But let's be honest, mostly i used it as my *motivational* partner, keeping me on and telling me how nice i am for writing this.. sigh. And that is something i would really like to thank Gemini for :). Be it not that supportive, i'm not sure i could spend that much time to make it :).

Let's finish with a typical human EOS token:

*If you believe this text is more signal than noise, click a "star" on this github page, let it be my RLHF :).*

Enjoy your time of a day!

...Chapter 4: secret chapter...


© Drazdra, 2025, Licensed under CC BY-NC 4.0
:)
