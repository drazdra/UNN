# UNN
Understanding neural networks

## Preface
Most people here don't know me, as i mainly live in Ollama discord server, under the nick "Drazdra". That is the easiest way to contact me. I rarely visit github (may do several times a year). 

#### First warning
This is top abstract stuff, if you are not into it, don't go further. There is no empirical research data or code.

#### Second warning
This field develops too fast, so i might not know a lot of already existing researches as i don't read these and don't work on inference engines.

#### Third warning
Forgive me if it's all too simple or naive for you :). The truth is.. that was my goal :).

Content:
1. A very abstract understanding of what neural networks are in general.
2. An overview of the way transformers made conceptually, easy for understanding
3. Several of my own ideas on how to make AGI :).
4. ...

#### Preface.
This text is just an attempt to explain things about neural networks and transformers without any confusing ML slang, without any math, without any code. It doesn't mean it's shallow or simplified, more like the opposite.

Its goal is to make you understand what's actually happening and why/what for, without having any special knowledge in the field. Transformers are pretty simple if you understand what they actually do :).

I recommend to read the whole thing before getting back at me :). 

So, if you like neural networks..

# Chapter 1.
## Neural networks and inference: ontological methodology.

I'm sure everyone here knows how neural networks work.. yet please allow me to share my own way of seeing these, as i believe my view provides a clear, intuitive and effective way to see and engineer neural networks. After all, i was arguing about this for years now :).

Yes, i oversimplify certain things but not to the point of making them wrong *conceptually*.

Actually, the whole point here is moving up from the implementation plane to the conceptual plane and to make it as abstract as possible.

We can have any amount of implementations but *conceptually* all neural network are still the same :).

Let's start with the most naive question ever, that nearly everyone can answer: how neural networks differ from regular software? 

To answer that, i suggest seeing neural network as a "mind" *of a kind* :). But please, don't get triggered here, i don't mean it's a *human* mind :). But it's certainly a mind as it can *analyze and predict*.

Yes, it's most primitive type of analysis and it processes data not the way we do, but it still can do both. It can analyze data to detect where the cat is and where the cup is and it can predict words well enough even to write essays, it even creates videos now predicting next frames.

To create this mind, we, as humans, code only its brain structure and functions. While the actual content of the mind gets structured on itself, automatically, with the help of these structures and functions we have created. 

And, *for now*, it's the same as it is with humans: we just can't add skills or knowledge right in. We can not take a person and transfer our ready skills or knowledge right into their mind. 

In other words, in fact we just can't teach anybody anything :).

What we *can* do is to provide the *conditions (including the information)* for the mind to learn. 

And even then it may still not end well.. *like me and solfeggio* :).

So what's teaching then? Teaching is merely creating the *conditions* where one has higher chances to *form structures* that embed new understanding and new skills. 

When you "learn" a foreign language, your brain forms new structures. When you "learn" playing guitar, your brain forms new patterns for controlling your fingers and for new type of hearing.

Why the "learn" in quotation marks? For the same very reason! 

Can you even say it's *you* learning it? Not really, you just spend time trying something and then *something* emerges. Or does not emerge :).

You do not patch the connections in your brain manually, they just emerge somehow, if you try hard.

It's the hidden work of internal implementation of the brain structures and functions, that we don't either control or feel. We just get new patterns created in our brain that we can use when they are there.

Same goes for neural networks, they don't control *training*, it just forms some new patterns inside with the help of algorithms we have created for the goal.

One more time: it's all about just getting new patterns somehow created inside of the info space. That's how "learning" happens.

Patterns capable of new functions. Emerging capabilities.

### But what do these patterns reflect? 

It's easy, just think how models get trained: we feed them with an ordered list of repeatable elements (patterns) and we want them to reproduce something specific in result - certain wanted patterns.

So, what can they do to achieve that? Of course they simply have to learn the mutual compatibility of all these elements we show to them. How every element goes with every other one, and what are the chances for it to go there depending on what we already have in our pattern.

Isn't it simple?

Let's rephrase it: neural networks form patterns that reflect the *chances* for every data element to be around any other element.. and that depending on other elements around (which we call: the context).

Or, you can say they reflect how closely certain elements (pieces of data) relate to each other, depending on the surrounding elements.

Think about taste for fashion :). It is not about specific fashion pieces, it's about feeling what goes well with what, depending on the whole ensemble and surroundings. See? Simple.

And that's it. Nothing else. It does NOT store any facts or any information. It doesn't have any texts inside, it doesn't do "cryptography" of your information. There is no text, no images, no videos, no sounds inside. It's not a fancy encoding of your information as such, it doesn't encode any single data elements sequence.

What it has, is merely a huge amount of reflected patterns, embodying a cloud of relations probabilities. 

And that for all the elements' configurations it saw - for all patterns.

In other words, neural networks do not reflect the data itself the way storage systems do. Neural networks reflect how data is organized, the possible relative distribution of its elements.

For example:
How closely kitten is related to cute and to milk. How closely snow is related to cold and blanket. 
And how closely blanket is related to coziness and coziness to a cat :). 
And that is within the context of winter and within the context of underwater swimming.

But don't get me wrong, neural networks can go beyond just human ideas or words. They can process any data elements - sounds, syllables, numbers, images, weather, galaxies, anything.

Based on this, let's write down what defines any neural network architecture, what makes it *to be* one:
 - how they represent (store) these patterns 
 - how they actually find/match these patterns 
 - how they process these patterns (inference, classification, distillation, etc)
 - how they update these patterns

If you go down one abstract level and define how these should be structured and work, you get a specific neural network architecture.

If you go down one more abstract level, you implement these structures in a certain form, like in code or with a physical device and get a specific model/engine pair.

But at the top level, all of *machine learning is simply creating these patterns*. 

Inference is simply an algorithm that can *continue* the pattern, by finding the best fitting continuation based on all the patterns it knows.

Classification is simply finding the best fitting pattern within stored ones.

And memory, learning and thinking are *matching*+*processing*+*updating* the patterns.

Very simple, isn't it. It's all about beautiful patterns :). 

Before moving on, let's devote a bit more attention to the matching part as it has a trick up its sleeve :).

That trick is in treating the existing context as a specific *point of view* to find the best matching one. So that within a single huge pattern we can find/pack various multiple patterns, depending on the perspective we take to look at it.

You can visualize it as a starry sky. When a neural network gets trained it creates a starry sky pattern.
You can see the constellations forming certain specific patterns but if you move to another star system, same stars now form a different pattern.

Seeing multiple patterns within one single pattern is just a matter of perspective or, in other words, of the context.

Now, let's move on.

### How can it store and process all these probabilities? 

it just uses a hell lot of conditional "if/then/else" statements.. 

Sorry, i just couldn't resist making this joke :). It's not literally the case of course. 

Mostly, in the modern neural networks we just use the math instead of actual code branching/logical rules, etc. 

The patterns inside are represented as numeric values, so the math operations naturally process these patterns and everything just flows natively. By "comparing" these numbers, neural networks can find how similar the patterns are, by "adding up" numbers they can just mingle any patterns together. 

In a way, you can see it as merging two images within your image editor. The image files on the inside are just lists of numbers that represent the brightness of the every dot, so we can simply add up values from both files to get a new image - a mingled image uniting both patterns. 

In a way, neural networks do the same, they just know how to properly merge numbers to get a new pattern or to match these and find how similar these are.

(we still may have certain hard conditions like in ReLU or in samplers but more about it later)

Once again:

a) Neural networks usually do not operate by stuff like true or false but rather work with a floating values whose "trueness" evaluates within context my "matching similarity" and most of the time it just changes the existing state (changing patterns), not choosing between different code blocks. Patterns just merge naturally through mathematical language.

b) There is a huge amount of the values in neural networks, which makes it possible to create an immense pattern and to store immense amount of possible *perspectives*, to see any required part of this pattern, to match the best fitting part.
(to repeat once again, it's a pattern of probabilities of what goes well with what)

c) Any value in the "dimensions" can work as one more perspective to the same pattern, it can be used to find a required sub-pattern, serving as a "branching condition" to the concepts extraction.

This is a very hard thing to grasp (high-dimensional geometry in the semantic space) so i will explain it in a way that anybody can understand :). 

> How a single value in a single dimension can have a drastic decisive effect on a pattern consisting of 10000000 other values? Just imagine a picture of a statue in a textbook. It's a beautiful piece of ancient work, it has tons of dimensions - weight, color, material, age, complex shape, history, art school, author, etc. And now, a bad student takes a pencil and draws upon it.. well.. let's say a mustache. The whole thing suddenly changes :). That's how a single new value can affect the whole big concept and its perception.

> Or, imagine a great tasty nice looking fruit - it's a good food category. Now just add a worm to it - and the category is all different now :).

d) neural networks can have multiple separate "big" patterns with sub-patterns.

e) the way neural networks process patterns is non-linear, which is a fancy way to say that one more single tiny dot in a pattern can cause an avalanche and taifun at once! So, *in a way*, it *does* have these if/then/else blocks.. *runs away giggling*

Under the hood, we may have scary matrix multiplications, sigmoid functions, normalizations and so on, yet conceptually they all just serve as means to match the patterns and to mingle the patterns :).

So you can change sigmoid function to something else, you can replace dot-product with qubits, you can do lots of other things but conceptually it will be the same - implementing the pattern matching and growing it further according to the rules of our learnt pattern :).

### How does it know which patterns are there and which do match?

Modern training is not really intelligent. Usually, during the training our functions just try to change the values in the neural network's structures, until they finally amend the internal pattern so that it can reproduce the new pattern you gave it.

The check up here is if neural network can match (detect context) and continue the pattern properly, giving you a fitting reply. After all, the reply is just a pattern made of words :).

Under the hood we usually just use gradient descent algo that tries to find which values and how to change to reflect your own pattern the best. 

Getting back to a starry sky example, we can see it as changing the stars position so that when viewed from the correct viewpoint (context), we can see only the fitting neighboring stars at the right distances, giving us only the fitting possible constellation that can tell us which star comes next.

Of course, for every *next* desired output we still may change *the same* values we changed for *previous* desired output.. Which means that whenever model learns something new, it may distort whatever it learnt before.

And that's exactly why training is lossy by definition. We use the same parameters to reflect multiple patterns, imprinting into a single wooden frame immense amount of sub-patterns, which can be seen from various angles or in a different light.

And this is basically how the whole training works.
(I omit here various tricks helping models to reproduce the pattern faster than just by super slow trying of all possible combinations)

So, again, neural networks do not store the data you show it, it stores only the approximation of your data structure, the internal organization of your data, not your data itself. And to do it, it forms patterns..

Which means - neural network is a *pattern processor* :).

And that's exactly why we can use it for understanding and development of *anything*, apart from pure Chaos :).

Because understanding something means *being able to predict* it. 

To understand something we first analyze it by trying to grasp the specific pattern of the process and then we correctly predict that very pattern's development over time, which proves we have grasped the pattern adequately. 

If we can not predict its continuation, it means we do *not* understand something that goes on, we don't see the whole pattern. 
(or there is no pattern and there is the pure Chaos factor).

That's why there is no difference for neural networks if it is DNA, sound, text, images, laws of physics, traffic, human's face or weather forecast. The way they work can grasp any patterns in any system all the same way.

### So, what does this understanding gives us?

It provides a basic concepts for building a good AI.
1. It should allow forming the patterns in the fastest, easiest and most flexible manner.
2. It should allow forming the meta patterns - patterns for the built patterns, with any amount of reflection levels.
3. It should allow rewriting the patterns - updating.
4. It should be able to match patterns (and find the perspectives).
5. Updates should be minimally functionally degrading for existing patterns.

This is valid for *both* ML part when it creates a reflection of data organization AND for inference, where it creates a prediction pattern.

Simple, isn't it. You want to have progress? Add dimensions, *add reflection layers*, add update mechanics.

# Chapter 2: explaining transformers with ponies.
Now, let's do some understanding of what transformers are. We will talk about decoder only transformers that we all use now for text generation.

In this chapter we go down one level of abstraction to talk about a specific neural network architecture. It is going to be a bit less abstract and to have a bit more of actual architecture implementation specifics. But not to the level of actual engines, implementing the architecture.

First, i believe the term transformers itself might sound fun and is right in technical terms, but it doesn't really help much to understand what they do.

What i would actually call "transformers" is _incremental associative morphers_. 

Doesn't sound that cool? :) 
Yet this naming is way closer to what they do on a high level and less confusing. To me, at least, and i believe soon for you.

So, let's unpack it a bit.

Transformers have 3 main conceptual blocks:
1. Input
2. Copy-pasted group of certain layers repeating over and over - "repeating" blocks
3. Output

These blocks play different roles and we shall take a good look at each of them.

### Part 1 - Input block
The role of the input block is to translate your words into the "brain signals" of the model.

Input just translates each element of the text we send into the patterns your model knows. These patterns define the place of the text element you sent in the grand pattern model has. 

It's the internal *interpretation* of our data the model has learnt. It allows model to *relate* your text to everything else it saw and to find something probable to it, something that can *happen next* in your text.

In text transformers these input patterns are limited to a *fixed* list after learning. So the model has no access to the "raw" sensory data anymore - it can't see anything *new* it didn't see during the training. You can't send a word in some alphabet the model doesn't know. You can not send an image of hand written text. It will just have no idea what it means in the grand pattern, where its place, what goes next. 

Does it mean that input should *always* be split into fixed pre-learnt patterns? 

Nope. For example, for images and sounds it can be different. A brighter image can make the input pattern have certain values be higher. This way model doesn't have to see the every possible combination of brightness at training, to learn how to understand the "brightness". The input becomes variable and the model can still match it because it's still similar to what it had learnt. And so it can react to the raw data variations, instead of just having a fixed list of patterns.

#### How do they learn to convert text into.. patterns?
During the training, neural networks learn to represent every possible element we send to them as a specific internal pattern, recorded as a long list of numeric values. Just the way image files are stored as numbers on the inside.

Every single "row" of such values is called a *vector*. We call it a vector simply because of the way we deal with its numbers - how we add them and compare. But as this is not a mathematical text, i will just use the word *row*, and for its content i will say values, and for their position in the list i will use *axes/dimensions*.

A table of these is usually called "matrix" and a single "book" of many matrices is called "tensor", but i will just say "matrix*.

And *pattern* here is any single numeric representation, be it a vector, a matrix or a multidimensional tensor.

Now, when i've triggered serious ML scientists and they have closed the page, let's go on :).

So we have a text to process and we need to split it into a list of basic elements. 

Could we split it per character? Yes, but in the current architecture it produces so many combinations that it becomes too slow and expensive to use. One token in average equals to 3-4 characters, so be it 1 character the text generation would slow down for at least 3-4 times. And i don't even mention the training costs! Just imagine how many more combinations it would create!

Then, could we split it by words? Nope. Words in many languages often morph, so it is not an efficient way - too many of them. Also, we have other things in text, like numbers.. 

And we decide to do something in-between - to split the text into a most common combinations of chars. That is:
 - common word parts: hi, under, sta, pro, num, li, etc
 - static common words: Hi/Hello/hello/HELLO/car/etc
 - common numbers 1.. 9, 13, 1111, 12345, etc.
 - punctuation
 - emojis
 - and so on

Each of these basic data elements is called: token.

Note, i made examples for text only transformers, but in fact it can be *any* type of data: images (where tokens are typical combination of pixels), audio (typical frequencies combinations), motion (typical coordinate changes) and so on.

### How is this list of tokens created?
Before the training, a special software takes all the training data and splits it up into the smallest list of basic elements. So if our whole training consists of the words "Milk" and "Cat", we get only two tokens: "Milk" and "Cat". But if our training dataset also has "Catie" and "Milkie", our tokens list would get a third token: "ie". 

On the inside, every token consists of a personal list of numbers, where every number has its position (#1, #2, etc). Every position (column) works as a separate axis or, in other words, as a dimension. Since every token has the same amount of values, all tokens together make a single table (matrix) where every row describes one token. 

If we take all of these tokens of a neural network together, they make *the vocabulary* of a neural network. A list of all elements it knows, even tho most of this list is not *our* human words but the text chunks.

A numeric list representation of a token is called "embedding". Further, when i say "token" it usually means "embedding". But to avoid complicating the story i will just say "tokens". Why? Because "embedding" is just an *internal representation* of a token.

When you send your text, it gets split into these tokens. Then neural network looks up a tied list of numbers in its vocabulary for every token and just replaces the text with numbers. It puts every token onto a separate row, so instead of a block of text we get a table/matrix where every row is a single token of our text, represented as a row of numbers.

You can visualize every column (position) of rows as a separate *axis* of the token's pattern. Like, first number in the row represents the horizontal coordinate, second number is vertical coordinate and so on. 

But it has so many columns! Like thousands! How to imagine that?!

That's easy. To imagine more dimensions just think about pages in a book where each new page carries a subpattern of just 1 or 2 axes. Then a single whole pattern is a *whole* book of these sub-pattern pages. So every token is a book of sub-patterns.

How many axes/dimensions/pages these books have? Just as many as developers decide to give it. It's something they decide upon before training, as the more axes you have, more time you spend to train the model, more money you need to pay.

#### What do these columns/axes in the tokens actually mean? How are they used?
And here i could go an easy route and say that every axis has a specific meaning. Like the first axis represents everything "white", second axis groups everything "curvy", another one can mark everything related to "name" and so on. And if you have a high value in the first position for some word/token, it would mean it relates strongly to the "whiteness", while the "black" at the first position would have a very low value, showing how far it is from the "white". 

But.. it would not really be true :). It's actually a *very* misleading way to explain. And.. just wrong.. :)

Let's dig in why.

#### What does neural network learn during the training?
It just learns the chance for tokens to happen around each other.

To do that it has to form certain ways to *relate* tokens. Of course some of the tokens come together more *often* than others. So they are *more related*.

How to record this relatedness? 
Of course through *similar* patterns of the tokens. 

Neural networks do it at training by finding the right values for all token axes and other parameters. The goal is to make certain tokens look like a "star" and some others like a "circle" :). In a figurative way.. So it's all about figures in patterns, you see :)

And the beauty here is that it doesn't matter how *big* these stars are or how *small*, you still see the *stars*, not circles :). You still can match one "star" token to any other "star" token and find the resemblance.

That means that *actual* numeric values created at training are not *that* important for classification. Most important are the *proportions* between values, as they form a figure in the pattern. For convenience, we can imagine some proportions of these values as a multidimensional "stars", "circles" or even as "pizza without a slice" :).

Once the learning is done, neural network can then compare these token patterns and see how compatible they are. 

Do not think that every "star"/"pizza"/"circle" patern is a separate isolated group. No, neural networks have a way to measure how *close* the pattern is to every other token and to decide on its similarity based on that. So we don't really have *groups*, instead we have a way to measure the chance of a token coming based on *similarity level*. The circle pattern is closer to the pizza pattern than to a star. And each token's pattern has multiple such figures inside.

Until learning happens, nobody knows which token groupings are to be reflected, nor which patterns neural network will create to unite them. But after the learning, the figures represent chances for every token to be around every other token and their *similarity* forms stable token combinations.

It's like now you know which people will agree to come to a party, depending on who comes, but for the whole city! :)

In simple words, certain proportions in the token values reflect that tokens a, b, c have higher chances to come together but won't come if token "d" is there as its pattern breaks the compatibility.

One more example: say you have tokens with low values on pages 16 and 17 but high values on pages 5, 12 and 121 - that is a sign that the token *might be* *at least* related to a "fairytale" related tokens cloud. That's because these tokens always went together around the fairytales context in that very text that model was trained upon, and the model has found a way to reflect it through this "figure" within of its patterns. But always remember that the same token can still be related to many other clouds of tokens. And also a change in any of the axes may suddenly make it related to a totaly different clouds of tokens, as it will turn existing figures into something new in the same pattern.

An *insight* here is that every token, in fact, is *not* a fixed symbol on the inside. Every token itself is just *a cloud of traits*, that can be aligned with *other clouds of traits* - tokens. Every token is *a concept*.

And yes, these multiple similar traits encode multiple clouds of related tokens :). 

So there is no "chars" on the inside, no "words", no facts of any kind. There is *just* a pattern that can be related to other patterns. 

How do most people interpret this? 
"If the words are tied sensibly, it must mean it has ideas! It stores the facts! It has knowledge!" :)

Sounds fun, but.. what it stores is just chances for data to be organized in some way :). 

But i said it's a concept.. Concepts? Ideas? What a concept even is?

"Concept" means a combination of basic elements into a single new basic compound element. A concept car is a car built of a new set of basic components: electric engine, propellers, carbon, steam machine, ai, paws, etc. In our case we speak of thinking, so it's about data elements, and for the language it's words. When we *think in language* we operate by meaningful semantic units. We can build concepts based on functional understanding, on traits, on abstractions, etc expressed *in words*. A concept of "comfort" includes a lot of basic ideas - where every idea is expressed in a certain word - warm, convenient, safe, relaxing, pleasurable, etc. And we can think about its *words* as its compound elements - basic concepts making a new concept.

But transformers form *concepts* based on mere symbols sequences grouping! So the concept of "comfort" might be additionally tied for them to "concepts" of "like ", "ed", "ing", "car", "dis" and even to the comma ",". 

They don't have *any* other experience than text.. they unite into concepts the grouping statistics of chunks left after the tokenizer.

*Transformers *lack* a whole necessary layer that makes humans humans - language representation.*

We do not relate the *semantic units* in neural networks, we relate altogether both: meaningless chunks and the proper words that carry the meaning as we mean it :). 

That is, transformers on the raw level often do not manipulate on concepts made of *semantically meaningful ideas*, they work with *concepts* made of tokens which may show mere statistical distribution of the *chars*, not human *ideas*. It's like humans *partially* trying to make sentences based on most common *sounds* pronounced together, not just on *words*.

That's why tokens are a paradox, on one hand neural networks have to use these to create the clouds of related tokens - to "conceptualize" the data. But on the other hand these tokens often are *not* meaningful as we mean it..

..and something tells me, the guy who invented tokenization was a programmer :). It perfectly *optimizes* the encoding of any language and at the same time *muddles up* the whole conceptual purpose of doing it :).

> *..or maybe the person just didn't tell us the whole system? more about it later*

And at that, neural models manage to do the trick of imitating humanlike *concepts*. How?! Why? Three things:
 - tokenizers also have a lot of full words and word roots, that are *our* concepts
 - the attention implementation doesn't process learnt patterns *separately* but rather always mingles these into a single big pattern, working as a *semantic constructor*
 - the amount of statistical data is immense, making it learn to translate their conceptualization into our one (remember, transformers started as *translation* neural networks)

These three things partially patch the initial "flaw".

Now, when we see how "concepts" emerge in the transformers (as a typical traits clouds - tokens) let's take a look at the classic example with the "King" and "Queen" tokens. 

There was a talk that if we apply the pattern difference between "King" and "Qween" to the "Man" token, we somehow get a "Woman" token. Or vice versa. And the conclusion that it must mean this specific pattern change (vector shift) encodes the "gender" information :).

However, it's not. It's because "Man" pattern is *already* associated with a certain cloud of related ideas (man, dirty, socks) and has a certain shape. By introducing these specific pattern *changes* *to that specific* shape we make it partially closer to that *other* queen cloud (woman, pretty, stockings) and more distant from certain tokens of its original cloud (man, stinky). 

It may seem we encode "gender" in these changed proportions but in fact we just change the strength of ties to other *multiple* tokens by amending the existing pattern. And weren't these patterns already *similar* or.. *compatible*? :) And the *change* we introduce may in fact bring way more than just gender. Potentially, it can change relatedness to *all* of the tokens. 

More than that, if we apply the same change to some irrelevant token (with a very different pattern), the change may not introduce anything gender specific at all because resulting related figure still will be very different. It would just move it closer to some different cloud of related tokens
 ..*and who knows what dragons live there!*

Summarizing it all, this is how training creates patterns that capture tokens' mutual relatedness in their probable combinations. 

Once training is finished, all these token/patterns are "frozen" and neural network only reads this learnt vocabulary but never changes it. Whatever text we send, it immediately *knows* which of the tokens/words go well together next and which do not. It can compare, it can match, it can mingle - all because the text elements are just patterns written in one internal "language". 

And for the same reason it can easily translate back into human words any *new* mingled pattern. It just finds a most similar pattern in the vocabulary and looks up an associated token. Then it just takes the word/chunk of that token and prints it back for us. And yes, note the "*A*", as there are always multiple similar patterns/tokens :). But more about that later.

Last thing is, what if we send to neural network a long text consisting of random characters? No template, just that.

In that case our model will just *continue* this list of lowly related meaningless garbage, sticking to this weird pattern we have made. Because the mingled pattern will be super noisy and won't have any figures a model could detect, so it won't fall into any sensible pattern. It will match only itself, and the continuation will reflect its own configuration, not the learnt probable patterns.

Until.. it finally stumbles upon some familiar pattern. And then it will just continue it, forming sensible sentences. 

For example, if you end your long garbage string with a single question mark, that alone might be enough to make model continue with a sensible text. That is because:
 - garbage = "weird pattern not matching anything from the common sensible patterns, not even to a garbage as all garbage examples look different. it matches only its own parts"
 - ? = "a known pattern related to a typical normal reply"
 - "garbage+?" = "tokens typically coming as a reply when nothing else familiar detected". 

If you send a typical *templated* request with garbage, the neural network will generate a sensible reply right away, because the template tokens will work in the same way as "?", model was already trained how to reply when the only detectable pattern is the template pattern and some noise.

And upon this we finish the "input" block. 

I think this part was pretty easy, wasn't it? :).

#### Part 2 - repeating blocks
This is the core of the transformers, so it will be the longest part with loads of sub parts. Let's start.

It consists of a repeating structure, where every repeated block consists of the same set of layers, yet with a different content. As if somebody copy-pasted one block many times before the training and they all now hold different patterns but structurally the same.

The way they work, is like a one way conveyor where every next block does its specific changes to the pattern.

Each of these repeated blocks of layers is "isolated" from the other ones, they do not cross talk in any way, they process data consecuitively, a block after block.

We can repeat this whole conveyor many times, each time creating a one more new pattern/token, adding these to the original text, making the text/pattern longer and longer.

We call this process: inference - creating a new pattern (token) from the existing pattern (tokens). 

In a way, you can see the work of transformers as a line of people. 

1. First guy splits your request into chunks and writes your request on a piece of paper using their internal pattern language. Then "adds" a space for the new part of the pattern.

2. Every next person in the line (repeating block) interprets the input pattern and mingles its summary into the new part. The original text also changes as the person also mingles-in one's own new interpretation. Then the person passes the paper to the next person and so on.

3. The last guy takes this paper and translates only that very last pattern part they made to a human language. Adds it to the original human language text and.. feeds the paper back to the first guy, to translate it back into patterns..

4. The only thing kept between these cycles is the text interpretation, so from the second cycle people do not interpret the existing text, they reuse their own past interpretation and work only on the newly added part and the one to create. That's called kv cache.

5. Whole operation repeats from the step 1. 

I mean, even if you tried, it's hard to make things more weird. Doesn't it remind you a telephone game? :)

They were trained together to produce certain result but they never talk to each other, they can't go back, they do not have a separate paper to do calculations or thinking, etc. They have no plan. They just mingle-in the very first associations they have to all the text they have by now, on a token by token basis.

#### But how exactly can they interpret the data? 

Spoiler: the whole approach to the implentation feels very much like "let's just put several more similar blocks and see what happens" :).

Let's talk about the internal parts of the repeating blocks now. Every one of these consists of 2 main things:
 1. Attention block.

    It has one O matrix and multiple identical sub-blocks consisting of their own Q, K and V matrices. Every of these identical sub-blocks works separately in parallel on the same input and is called: "attention head".

    V matrix per attention head learns a rough idea of how to extract useful data from token patterns, so their traits could be mingled together in a unique way, without adding too much distortion. In other words, how to interpret tokens.
    
    Q and K matrices during the training learn to decide upon the compatibility of the tokens with the attention head, which results in measuring "relatedness" of tokens to each other. 
    
    O matrix stands above these and knows how to extract useful data from the *united results* of all attention heads. In other words, it knows how to create a new single pattern from separate smaller patterns produced by individual attention heads. This way after the whole attention block we have a single representation of data. It also tries to filter some noise produced by attention heads.

    Whole attention block produces an *overlay* pattern that is blended later into the original token once the attention is processed. This overlay is a *change* to our old pattern and doesn't have to contain all of the information required for a full token pattern.
 
 2. The FFN block, goes after the attention block.
    
    This block learns to "*fix*" things after we add attention results to the original token :). It works like typization in a way, with the difference that it produces *changes* that should be applied to the token as it comes from the attention block.

    So it's like typization *to a degree* that can introduce new and dampen existing traits of the new token's pattern, making it better matching the fitting typical token clouds.

    While attention infuses traits of past tokens into all tokens, this block makes every separate token more *identifiable*.

#### Starting with the attention block: Q and K.

As explained above, on start we split the data into tokens, translate these with the input block and go over it.

So we have whole context represented as a table where every row is a token, and we start by sending it to the first repeating block's attention. Literally, the whole table goes to each of the attention heads in there.

Every of the attention heads has its own single Q, K and V matrices it has learnt during the training. Basically, that's what attention head is: Q, K and V matrices and some logic of using these.

The goal of attention is to find all *related* tokens in our context and to fuse their traits, so we could find something that is related *to all of them* together, that is to their context. By idea, we could just try to combine traits from *all* of the tokens together. But this is impossible as our token's pattern has a fixed length and can't encompass infinite information. Trying to add everything into one little thing would just ruin its internal structure, it would be a cacophony.

So we need to find only the tokens that actually often come together and so are related. Then we could unite their traits and see what else usually comes together - what else is from that cloud.

These Q and K during the training learn to *detect* certain specific traits that model sees as stable and relevant figures. So each attention head gets its own specialization - reacting to specific cloud(s) of tokens.

But more than that, both Q and K learn to express the result of their findings in a *compatible way*, so their results could be *compared*. 

This makes them be two parts of the same function, as only the *comparison* of their results is used in the system.

And the goal here is to find out which tokens are compatible, so we can mingle them, and how much we can mingle them without breaking the proportions.

Which tokens do we actually compare?
Well, of course all of them, we need to know which tokens this attention head can mingle, so we just go over every pair of tokens - comparing each next token to every preceeding token.. *and over itself too*. 

So a word chunk #2 is tested against #1 and.. #2, chunk #3 is tested against #1, #2 and.. #3, and so on for every pair.

> By the way, on ML slang this is called "causal" attention, as in "cause and effect", causal here is literary - "preceeding/determining the future". And all of that is done in batches, doing math simultaneously, to utilize full power of gpu parallel processing

Why we need to compare a token to *itself*? 
 - traits we extract *from* some token might break the proportions of the whole token, when added back (remember, in the end we mingle everything back into the original tokens)
 - we need to know how much to scale the traits of *all* mingled tokens relatively to each other

As the final purpose of comparison is mingling of tokens, in every pair we have a token we mingle-into and token we extract the traits from. Let's call them:
 - the token we mingle-into: a target token
 - the token we extract the traits from: a donor token.

So, how exactly do we compare them?

First thing to understand is that we use the "input" version of every token in the sense of "as it came" to the attention, *before* we changed it here. So the first repeating block's q/k see the tokens as they are from the input vocabulary, and every next repeating block's q/k already see the tokens after they were processed by all preceeding blocks.

Second thing to understand is that before comparing the tokens, every attention head goes over every token and creates its own Q, K and V *results*/vectors. These are created by multiplying the token content by the content of attention head's Q matrix, K matrix and V matrix. So we get 3 new different patterns/rows/vectors per each token, per each attention head. And each of these patterns is just a single row of numbers.
> i will explain "multiplication later in the ffn block, this section is too dense as it is

Attention heads care only about *their own* token "interpretations", their own created rows - they do not interchange their data at all.

So now, when we have this extra data per token, we can actually compare the tokens by these created patterns. 

Specifically for the comparison we need the patterns created by Q and K matrices. But.. which ones do we use to compare?

Whenever we compare two tokens, we always take:
 - Q row from the the target token
 - K row from the donor token

What do these rows actually *mean* conceptually? My own interpretation of this process is:
 - Q detects if the target is compatible with the extracted traits *this attention head typically produces*
 - K detects if the donor has traits compatible for extraction *by this attention head*  
 - And Q+K together learn how much to *scale* tokens' extracted traits, relatively to the target token

So if they are not compatible in some way, the extracted traits are just scaled to near zero and ignored.

Which *traits* every attention head actually detects? We don't really know, it just automatically happens during the training. Neural network just has to develop some way to relate most common token clouds by some traits and this makes attention heads specialize on some of these. 

But let me make myself clear here, i believe that Q and K don't just project two tokens *to compare them through tokens "distilled" traits*. It's not about intersection of *tokens*.

Q and K project "validation" pattern showing if this token is compatible with the *attention head*. Like a green lamp showing this token *can* be processed with *this* attention head. Q lights the green lamp if token can *accept* traits this attention head usually extracts. And K green lamp lights if token *has* traits this attention head extracts. So they are "compared" against *attention head* in the first hand and then the results of these two detections is what we actually compare.

If attention head could detect both:
 - compatibility of the target with the common extracted traits by this head
 - compatible traits in the donor with this head's extraction

the matrices learn to produce *similarly shaped* results - green light.

So the similar *shape* expresses their compatibility, while the shapes combined *size* (brightness) expresses the *proportions* that mingled-in patterns should take.

> Shape here means that their comparison produces a positive number and size/magnitude is how big this number is. We will talk about it soon.

How can it even work? Well, the thing is, that these 2 matrices are all tied to the work of the third matrix - V one. The one that actually learns how to extract certain traits from the tokens. And so our "green light" is the emergent feature of all *3 matrices trained together*: Q, K and V. They learn to be *compatible* with each other through being *trained together*. That's how they develop their specialization.

But.. why we need *two* matrices to detect the compatibility of the token with V matrix? Why couldn't we detect it with just one matrix?

The thing is, Q and K quite well can look for *different traits* in the tokens, because we may mingle here *different* clouds of tokens. Yet, the matrices produce results (green light) that should be *similar* for mingling to happen, despite *what* they *look at*. 

> Being compatible with traits *as a target* doesn't mean having the traits as a donor and vice versa. Even more, extracted traits from A token can fit B fine, but extracted traits from B token might *not* fit A token. This thing is not symmetrical, and we look for *different compatibilities* here. 

And considering that they look for potentially different things, yet their results should match in comparison,  they just *have* to abstract their findings into a different resulting pattern language *common for both matrices* - "green lamp". 

In that language they need to retell two things:
 - if they succeeded or not to find the right traits in the tokens
 - how to scale the traits to avoid distortions

Suddenly, relatedness of tokens becomes a level of compatibility with the attention head's "preferences" :).

And here you can exhale, as this is the whole concept of "relatedness" :). I've turned everything upside down here to explain, but i think it worked well :).

Now, let's go into more gore details and see *how* exactly we compare. You may safely skip this section.

As i said above, after comparing Q and K resulting rows we end up with a *single number*. Yep, it all goes down to just one numeric value that we use to scale the traits (V results).

But how exactly do we get this single number? 

We do it with a method called *dot-product* comparison. I wonder who makes these names. Whatever. So, to do it we just:
 - put Q resulting row of numbers over K resulting row of numbers
 - multiply overlapping numbers of two layers
 - sum the resulting numbers up

In a pattern way of thinking, we just multiple symmetrical rays of two patterns and then add up their final lengths.

If the value we have got is high, it means that patterns are compatible - their pairs of rays mostly were pointing the same way - into a positive or negative direction. If not, it means this head can't merge them. Literally, here are 3 examples for a two "axed" patterns: 

```
   Q1   K1     Q2  K2   Result
 - 2  * 2   +  1 * 3  = 7  = match (green)
 - 2  * -2  +  1 * 3  = -1 = do not match (red)
 - -2 * -2  +  1 * 3  = 7  = match (green)
```
Everything below zero we turn into a very small positive value, so after scaling the traits of a donor token that produced negative result are turned miniscule and don't change anything in the amalgamation.

Everything positive we first proportionally reduce (to get smaller numbers) and then normalize to fit the >0 and <1 range, so our comparison can scale the traits down only. 

Basically, that's all of the explanation on how we compare the shapes (sign of all axes) and the magnitude (final number value).

> The question is - does Q or K matrice *always* produce more or less the same shape when detects a good compatibility? Or in different comparisons they may produce different shapes for compatible tokens, yet these shapes are simultaneously similar from both matrices? I did some web search and tho i didn't find any specific research data, it seems that yes, it's more correct than incorrect :). And in part i believe it's tied to RoPE that we will discuss later. But it's just a side interest that doesn't change much conceptually.

#### Unnecessary details about the actual comparison process that you can skip :)
 
Once we have summed up two vectors we do some more operations that do not change the meaning conceptually, but  make them more stable, convenient and relatable across all tokens.

First, we just scale down the final value by the amount of axes there were: divide the result by a square root of total axes number. We do it to avoid turning most values to ~zero after softmax. 

Second, we normalize it with softmax, which means we take the comparison results *of all token donors and our target* and use these to rank every comparison score against them all. Here we lose the original proportions of the scores, as this function is exponential.

How do we calculate softmax? 
For every token pair we take the constant value of 2.71828 (Euler's number) as a base and raise it to a power of the "reduced" comparison score: if the score is 6 we do 2.71828^6. 

Then, to avoid big numbers, we divide every "powered" result by the sum of all powered scores - thus converting every value into a "percentage". And that's how we get our softmaxed value in the range of 0-1.

The effect of this is that highest numbers are much much higher than anything even slightly smaller. It's like every little addition in high values lifts that score more and more faster and faster. And negative values become extremely small fractions.

So only the *most* compatible ones can really add to amalgamation, even a little less compatible ones are reduced to a very small effect. We even have to first scale them down to avoid total disappearance of any gradient.

#### End of unnecessary details :)


#### A moment of critique

When we do the comparison with dot-product, if a single column/axis is MUCH higher than other columns and matches its paired column in sign (both positive or negative), it might make the final result to have a high value *even* if all of the other columns have different signs but low values. This is a flaw of the used comparison method (dot product) and model *has* to learn to walk around it by finding the parameters combinations in Q and K matrices that make this situation very improbable. 

Theoretically a model could abuse for easy detection purpose certain specific axes in the underlying token values, by giving them high values as a trait, but these attempts i believe are wasted by later normalization block, so it should mostly abuse the q/k matrices in this regard.

Another thing is that the only way to encode incompatibility of the pattern here is to use the opposite signs for the values or zero values. Just different non-zero values of the same sign can define only the level of compatibility.

All of this is a pretty limited method to encode signal and my idea is that mostly it works because all it does is detecting compatibility, which is a very simple signal to encode. And, it actually *should be* a redundant signal.. but more about it later :).

The q/k thing looks *very* smart, probably the smartest thing in transformers, being also very fast in terms of compute. Yes, a better comparison would be not just compatibility to the head but between *actual traits*, but that would be *a totally different attention story*.

#### end of critique

#### A moment of thinking

Here we get an interesting side conclusion that q/k may learn not just the level of tokens compatibility (critically low score category vs everything else) but also learns to work as a *patcher* to change the size of the patterns so their amalgamation would end up in a right proportions. This is interesting as it then learns to encompass two different functions: detect tokens relatedness/compatibility and to patch the size of mingled patterns for better mingling. Of course at that mingled traits do not have to be *the same* between tokens, as attention head may extract different ones from different tokens. But considering the amount of other things it should take into consideration, i would guess it's a rather weak emerging feature.

#### end of thinking


#### Distance between mingled tokens
If you have an inquiring mind, you may have spotted that now we know how compatible our tokens are, but we have no way of knowing how much they should affect each other, as older tokens should affect new tokens less than fresh ones. The "white" word should not affect all further words in the text as much as the right next one after it. 

How do we go about it? 
Well, to deal with it transformers introduce the RoPE trick. 

You can imagine it as an odometer - a device in cars that counts how much it has driven so far. It shows a number in miles or kilometers, where every number is on a separate rotating disk. Once the odometer reaches "0009" it turns into "0010" and so on. At that, all disks move *simultaneously* so it's not a sudden shift to +1 in the third disk. No, that disk just was slowly crawling there all the time and could be read as "half there" separately, when we had "5" on the last disk. 

This is exactly how RoPE works. It simply takes the position number of every token (milage within context) and converts it into pattern distortions in a *predictable way*. Literally.. it pretends that axes are these disks and rotates them.. 

How can it rotate an axis? Well, it can't rotate one axis :). 
But it:
 - takes a lined paper and puts a dot, calling it 0,0 coordinates
 - takes *two* axes values, pretends these are (x,y) coordinates and puts a second dot there
 - takes drawing compass, puts it into 0,0 dot and draws a circle through the second dot
 - and then with a weaked smile gradually changes these coordinates as if the second dot is alive and crawls upon that circle, until it returns to the same spot
 - and so on again and again

It doesn't stop at that and uses separate pairs of axes for various distances. So if the first pair is for, say, 10 miles/tokens, next one can be for 20 tokens, next one for 40 and so on. So at the token #40 the first pair of axes makes 4 full circles, while the last one makes only 1 full circle.

But.. axes of what? Which pattern do we torture this way? Well, these are applied to the Q and K results.

As i said above, the rotation angle depends on the absolute position of token in the context - their "milage" from the start of the context. The more distance is between tokens, the more of the axes pairs get rotated more.

If the original shapes were *similar*, once we increase distance between tokens, RoPE rotates thier axes and shapes lose their similarity and become *less* compatible. 

Can this rotation make different non-compatible shapes falsely compatible? 

Of course, when we increase distance between tokens the change between a few axes may suddenly get *more similar* by an accident. But usually it's not enough to make the *whole shape* similar enough because there is lots of other axes pairs that are still different. So an accidental "compatibility" in just one pair of the axes after rotation has very little chances of making the whole shapes suddenly false compatible.

So, is the RoPE perfect? Sigh. Let's see..

Once we do 180 degrees rotation due to distance between our tokens, vectors suddenly start *restoring* their original similarity, as if becoming more and more compatible, tho the distance just *grows*. 

This is pretty fun, however don't forget that all "disks" rotate, so when our first disk starts to restore similarity, our second disk already half way "distorted". So the overall compatibility is still diminished.

Another thing about RoPE is that due to implementation as our absolute positions of tokens in context become big, the rounding effects in the code may lead to precision loss, digital noise :). So in real world engine, the actual precision loss might intervene as we go further into context, due to quantization stuff. Be it linear logic it wouldn't matter much, but in our case we have non-linear effects where a little change might lead to big consequences, so this thing *may* at times affect results a lot.


A single attention head *may* lie about tokens "compatibility" score based on *where* tokens are in the context, even if their mutual distance didn't change. And it also can lie if the distance between tokens is out of its "safe" range.

And here we get another problem! The longer context we have, the larger amount of disks might lie about the compatibility. At the start only the first disks go outside of the safe zone fast, but later in context a lot of medium speed disks might also show false incompatibility.

So, let's try to predict some consequences of this fun, thanks to RoPE:
 - every attention head is applicable only within of a certain distance between tokens, otherwise it lies
 - at certain relative positions of tokens in context, attention head lies
 - to be able to still understand where there is noise and where signal, model has to *duplicate* attention heads with the same specialization *for various distance ranges*. So multiple heads catch the same traits just over different distance ranges
 - this redundancy is the only way for neural network later to understand which attention head lies and which ones are telling the truth

Now consider that the RoPE, once it is applied deep in the context, actually warps the original Q and K produced shapes a lot. If these shapes actually carried some complex single signal, there would be a really narrow range of distances where they could still keep both the signals and their similarity.

And.. we would hit the need for incredible redundancy in the attention heads, right? The training also would grow a lot in compare to non-rope solutions.

However.. it doesn't happen. It doesn't change *that* much. 

Why? Because the signal is very simple, it's not about matching the profile of specific traits of two tokens, it's just about compatibility to the head. 

And this one can be expressed easily in just a fraction of axes and... it can even be easily redundant across *different distance range axes/disks*. So that it's easy to develop a pattern that still stays compatible by just reusing abstact similar patterns over various disks that average each other. So when some go out of scope, other ones just become *more* compatible. And the compatibility signal survives. 

Of course it can't compensate for any range of distances, as starting positions of disks should be neutral, be they originally opposite, fast pair match wouldn't be decisive for the whole q/k comparison.

The RoPE still forces the models to create redundant heads for long distances and for short distances, to fight the false signal when multiple "disks" go out of phase and start to lie. But because of the "compatibility" trick it's much smaller redundancy than what could be if Q/K carried a really complex signal about the actual tokens traits.

Why do we still use the RoPE then?

Well, we could send the numeric position of tokens, but it doesn't work well because model gets used then that certain tokens play certain role at a specific position and we want it to be abstract as much as possible. We don't want the first produced token to be always "hi" just because it's the first one :). Also, this way it's much easier to *scale* the possible context length as model learns to encode any distance, not just a fixed range of positions. Instead, it learns to translate the distance into compatibility change, even tho with mistakes.

#### Now let's talk about the *user* side of the story

I hope you've skipped the previous internal mechanics explanation and now we can just speak about the human side of the story :).

For the easiness of understanding, let's pretend for a second that for pure random chance our network's attention head has learnt to match the traits of a specific idea of humans.. (it can't happen as transformers never do token clouds per human idea, they are based on tokens *sequences*, but we pretend).

Let's say our attention head may not care if the tokens are "color" related or "grammar" related. It cares only if the tokens are "politically" related! (sigh) So it's compatible only with tokens that have figures common for politically related tokens. Maybe the head was trained in 2025, who knows.

Then, on a higher level of abstracting, we can believe it measures:
 - relatedness of the donor token's interpretation (both are about politics and close enough)
 - significance of the related parts in the whole tokens (how much it is about politics)
 - significance of the related parts to the head's specifics (is the head about politics)

How we may think about attention heads then:
 - every attention head has its own interpretation of the tokens
 - when we test tokens for relatedness, we search for relatedness in the context of this interpretation
 - for testing we use the original token values *before* they are interpreted (as they entered this block)
 - because this way we also learn to see how significant the related part is in each token

But.. all of it is only in our own heads :). 

It's not what it is. In the neural network, these are just potentially emerging features that can sometimes happen. But all it does, is testing the tokens for compatibility level with the head's learnt patterns. And these patterns are based upon... mere sequences of word chunks.

Now let's take a look at the real world examples. And sorry, it's not going to be political :).

If our text is "white hare", q/k checks if they both are compatible with its processing and can be mingled, and how much. That is, if they are "related".

If it believes they are, neural network mingles some traits of "white" and "hare" tokens, making the result *closer* to a specific token clouds, let's say one of these clouds is "snow" related. 

What *exactly* our attention head specializes upon? Maybe upon any adjectives that change nouns? Or just upon the "snow" token cloud traits? Or something else? Who knows.. 

But our sum of "white+hare" suddenly is now related to the "snow" token, as:
 - "white" token often happens in the context of snow, and has traits of the "snow" token cloud
 - "hare" token happens in the context of "snow" and has these traits too

And as our attention head liked these traits, the snow related cloud of tokens becomes very much related to our new "white+hare" amalgamation :). We just made these traits much stronger. Of course, various attention heads also mingled tons of *other traits*, but we right now care about this one only.

Let's test it with Gemma2-9b (gemma-2-9b-it-Q4_K_M.gguf) (without template) on a standard settings in llama.cpp.

And guess what? We do not get "snow" as our next token *anywhere* in probabilities. Why? Just because this tokens combination doesn't *ever* end in "snow" in text. "White hare snow" text is improbable. 

What is probable? Judging by the output, most probable continuations are:
 - comma ","
 - new line: "\n"
 - "is".
 
So it wants to say either: "White hare," or "White hare is" or "White hare\n". 

That is exactly because of these *other traits* that other attention heads mingled in. 

These traits were much more prominent than "snow", because these words way more often end up with a comma or an english null verb. And that's how it works. Not by abstract "idea" closeness, but by the *statistical* distribution. So even a new line right after two words becomes a much more probable option than the "snow", which is nowhere around. Weird as it is, but traits of the "comma" already were embedded in the "white" and "hare". And they were more refined, stronger than the "snow" traits :).

But what about "snow"? Are the traits of that cloud even there, as i explained, or was i wrong? Is it encoded in the tokens patterns? To test it, lets nudge our neural network to produce a *surface* token. To achieve that, we just add "upon" word. "Upon" is almost always followed by some surface in texts, so it should work well. But which surface will we get? Grass? Floor? Asphalt? Clouds? Let's see:

White hare upon -> ```the snow,
A flash of white, a silent flow.
Across the field, a fleeting sight,``` (seed: 2366235071)

Ah, i forgot about "the"! But it's not important here, as you see the "snow" became a very probable continuation.

Now let's change it to "Brown"..

Brown hare upon -> ```the moorland,
A flash of white against the green,```

And as you see if we change to the "brown" we get the "moorland" as surface instead. 

So the "snow" traits are encoded in the "white hare" *even* when the next token is *not* snow.

..but the fun thing is that even for "brown hare" we still get "white" as well. How come, our hare is brown! 

Well, it's just because "hare" word often happens around "white" in texts, just as "brown" does. So we still got "white" traits in the mingled pattern.

Model wasn't trained enough to grasp that having "brown" before "hare" makes "white" *later* to be in this combination very improbable. There is no separate strong enough "brown hare" traits with their separate tokens cloud, so to say.

So "brown+hare" just didn't break attachment of "brown+hare" to the "white".. on the contrary, it made "white" closer, *as the "brown" is a color too*! And now we have "color" cloud traits in the pattern, making all colors  even more related, more probable.

And there you go! It added "*green*". Why? Just because of the above! *All* colors are pretty much tied together as they *often* are used together. And various different colors often happen in the *same* context. So they should share some similar traits too. That's how "green" later becomes more probable than most other words - we get tied to "colors" traits cloud more and more with every new color token. 

Of course i call it "colors" traits cloud only as a *simplification*, because in reality it includes everything typically coming *with* colors in text too. 

So the model is not "stupid" to mix up or forget the color of our hare, it just *works* this way - by pulling the probable *compatible* tokens, and compatible they are primarily by their common usage combinations, encoded in traits.

Let's play more. In European context the "white" is tightly related to the "purity" (snow, wedding ritual, innocence, etc). Let's test it too:

A hare of white embodies the -> ```moon, a symbol of purity and new beginnings.``` (seed: 1122261616)

A white hare embodies the -> ```essence of purity, innocence, and good luck. ``` (seed: 4173163411)

While making the examples i also got "springtime" in continuations and wondered why, it turned out the white hare is a symbol of spring in Japan :).

If we replace the "hare" with "rabbit" it goes right into the direction of "Alice in the wonderland" as "white rabbit" happens a lot in that story and neural network saw it at the training.

A white rabbit from -> ```the book Alice's Adventures in Wonderland.``` (seed: 2624672892)

But

A white hare from -> ```the Black Forest, Germany.``` (seed: 2800670882)

I'm sure you've noticed that this process is the very essence of being *biased*. Yes, as by definition it works by pulling the learnt *associations* together, basically it's a big bias machine.

What happens if we have unrelated tokens? Like, if our text is "I was swimming today. You look great". The "swimming" token pattern then isn't related *much* to the "look", because these are not happening much together. And transformer then doesn't mingle the "water" cloud much into the "look" token (or does it a bit). So the word "look" doesn't get *much* closer to whatever was associated with the "swimming". The traits of the swimming are not transferred to the "look". 

A thing to note here, as i said before:
 - when we mingle traits, we do not mingle the complete original tokens, we mingle only their interpretations (traits) created with V matrix that i will explain later
 - however, we do not just add or remove only the *relevant parts* of the pattern interpretations (traits) we mingle. We just blend these interpretation patterns (traits) completely, with all the irrelevant parts they have, mixing up everything. Of course it results in making certain features more prominent and makes some other features fade. Not to mention we change already existing subpatterns by this. In result we affect *everything*, making new token closer to something new, not just to the "relevant" part of the new.

However, as each attention head learns to extract its own type of pattern traits, it learns to minimize this flaw by extracting only more or less relevant pattern part for mingling.

And that is exactly what we do have the matrix V for.

#### V matrix.

This matrix is the final puzzle piece of our attention head triumvirate - Q/K/V, it learns to extract information about some specific traits the attention head specializes upon. 

Of course it is not just the copy of these traits from the donor token, it's a *different* pattern based on the input. It has different proportions, values and meaningful *figures*. We can say that if earlier in the tokens we had stars and circles, here we *may* have *ponies and elephants*. But as it actually *needs* information about the actual traits it extracts, it shouldn't be just some basic signal, the way it is with Q/K.

This "figures thing" consisting of traits is a tricky one to understand, so let me repeat everything again. 
Neural networks do not encode the meaning just per single axis of a pattern, they go one abstract level higher and encode meaning through the *combinations* of these. And not just with specific combinations of actual values in every axis, but also in the *proportion* of the values. Proportions of axes just make *figures*. Even more to that, it's not just figures, it's not just a pony, it's also how *skewed* it is. Which rays differ how. And so on. This way pattern proportions produce an additional new relation on top of just similar single values.

Relatedness of "white+hare" to the "snow" is not because there is some special single "snow" axis. It's *just because some of the new axes proportions in their full combination *relate* to the "snow" pattern*. 

And it's never a single "pony" or a "elephant", but rather a huge weird figure where every spike can mean something *when paired* with some other spikes or "flats" or whatever. 

Let's call it - a zoo! Or a *meta pattern zoo* within of our pattern. 

Like drawing butterflies.. that altogether form a cute face if we know how to look right.. and a pizza if to look at it the other way! 

And the V matrix role here is to learn how to find and extract the traits of some concept, recorded in both specific values ranges and proportions of axes values (traits).
 
In simple words, it learns to see a pony in the pattern, including how much the pony is skewed. And it doesn't matter if that pony rides on a turtle or not!

Now let's get back to what actually is going on.

As V matrix extracts the traits it knows per token, from a pair of tokens we get two rows of numbers. What do we do with these? We:
 - mutliply each number of these by the factor of "relatedness"/compatibility found by our q/k comparisons<br>
   That is for target token #2 and donor token #1:<br>
      &nbsp;&nbsp;V1 * comparison of Q2 and K1<br>
      &nbsp;&nbsp;V2 * comparison of Q2 and K2<br>
 - simply sum the resulting numbers of two rows axis by axis. Yes, we just put one list of numbers over another one and sum their values

And here we are - having an updated traits sum, a new figure. It has same same amount of axes, as we only summed the individual columns.

And that gives us our "mingled" pattern consisting of extracted traits from all the compatible tokens preceeding the target one and the target one itself.

Basically, here our single attention head can rest, as it has done its work upon every tokens pair. And all of the compatible traits were mingled into all compatible tokens :).


#### A bit of a critical thinking about attention block

 A dubious thing about attention is that we always draw a new pattern from the *morphed previous pattern*, instead of creating a really new pattern on the background of the old one, or correcting the existing pattern. 

It means that this implementation:
 - can not self reflect by definition
 - it can not work on a contrast as it always *mingles*
 - it can not come up with a totally different pattern/approach, as it always inherits and morphs, but can not change the context
 - it can not restart at some point producing an alternative pattern
 - it can not go back to some earlier point in context and to correct an incorrect mingling that poisons it by pulling in wrong associations
 
The existing pattern is always biasing everything. The existing context is always a part of the *new pattern*, they are mingled into a single representation. There is no memory of any separate specific isolated part of the context.


Another thing i would like you to note here that we encode the concept of traits "relatedness" in both: ranges of absolute numbers of axes values *and* in figures they form. 

Absolute values ranges *do* carry the relatedness concept because we mingle *scaled* patterns and system has to learn it as another measure of relatedness. If q/k pair returns *low* score, we multiply our V figure by it and make it *smaller* before mingling it into another token's figure. That means that neural network has to react to both: 
 - figures similarity expressed in proportions between different axes
 - the actual absolute ranges of values taken by the axes - figures size

For now just note that due to scaling we also use the figures absolute magnitude to encode information. We will talk about it more in the discussion of normalization step. 

#### end of the critique moment


And to finish up the attention block, let's repeat again that every repeating block (consisting of attention+FFN parts) has *multiple* such attention blocks (q/k/v) working in parallel, called attention heads. 

Each of them during training *hopefully* forms its own way of interpretation. This way they are *supposed* to match, tie up and mingle tokens *differently*: different tokens at different strength, extracting and mingling different figures from the patterns. 

And here things go weird, because *originally* each token had its single pattern/definition, but now each of the attention heads has produced its own *separate* pattern/definition per token! Each token now exists in multiple versions!

Do you think if we unite all these together we will get a longer list of numbers than it was on start?

Not really. Yes, every attention head gets a full width list of numbers per each token. But.. they produce way smaller list of numbers per token. That is with less values-axes, a narrower table. It's a simplified representation, in comparison to the original input, a shorter patterns.

What do we do with these multiple simplified versions of the same tokens?

We just take these resulting tables of tokens and.. stack their results together per token! So our table becomes much wider (than it was per head), but the amount of rows (tokens) is still the same.

And now it's as if each token was represented by its smaller reflections in multiple smaller mirrors - attention heads. This way we got a "faceted" view of each separate token. 

Of course in reality these "facets" have redundancy, as attention heads could grasp *similar* patterns and compare/interpret more or less the same things in the end. Making attention heads do something totally different is a separate task. Usually developers try various tricks in attempt to get rid of redundancy. Like initializiation of heads with different random noise, switching off some heads during training, splitting the tokens between heads by some rules, overlapping their attention and so on and so forth. *And here a single flexible attention head could be really nice..* :).

But remember, what i told you about transformers? It feels like somebody just liked to copy-paste a lot :). 

Also, earlier i was saying that the context is always fixed and we can't come with a totally new pattern. This copy-paste *patches* it by creating multiple interpretations, where things can differ. *Yet* they all still just *morph* the existing context, same pattern, they do not cross-talk and move only forward, even if they do it in different ways. 

So, what do we have to do with our faceted token representation?

#### Here we finally get to the last part of attention: O matrix.
Its purpose is to convert this faceted token view into *one more* different pattern, yet united. 

After all, it's transformers. If you want to do some operation, just transform everything into something else! :)

> And yes, of course every one of these transformations introduces some noise and loses some useful signal. So the more transformations we have, the harder it is to train a model. It just has to find its own similarities in every of the intermediate patterns. On the other hand, this is the actual way of how transformers work, their core instrumentary, allowing them to process patterns.

The role of the O matrix conceptually is to transform the faceted attention heads' output into a pattern that is compatible with the original pattern we had on attention input. 

Also, it tries to filter out single false attention heads results, relying upon results of all attention heads, thus patching the attention flaws. Remember what we talked of in the Q/K block? Here if the heads produced some redundancy information, O matrix can learn how to use it to extract the useful signal and to ignore the "lying" heads. 

And here we finally get the required *change* to our original pattern, that our attention block has produced. 

This change carries the shifts to original token patterns, turning them into a *different* clouds of traits. Clouds that now reflect *the sum* of their related traits per token. 

Phew!

#### Residual connection step
This is how we call adding of our changes to the *token patterns as we had at the input to this attention block*. 

It's literally just adding their values together. 

What's the point of doing it all this way? Why couldn't we just create an updated *ready* token in the attention block?

Well, having it this way adds certain uniformity to the structure of the pattern, our model has to adapt to the fact that its pattern figures structure should match the one it had on the input. 

By limiting the actual functionality of the attention to just the gradual changes of the original pattern, we resolve the potential warping of figures within attention.

The whole attention block in the end has to work as *fine tuning* of the original pattern, not as something totally new or free. And that makes it much less "heavy" and much cheaper to train, as now it has to find only the *changes*.

And mingling of the *simplified interpretations* of V matrices still works without making everything fall apart fast. The attention result just tunes the original pattern, *not replaces* it.

So, in a few words, here we return to stars and circles we had originally, but now somehow changed, hopefully reflecting their relatedness better, as every single token now includes the related traits of *all preceeding* related tokens.

And in the end of the attention..

#### Almost forgot.. the normalization block :). 
Usually normalization means making something less deviating ;). Not sure if this explanation helps, so lets dive into this block. It is much trickier than one might think, as it's not just scaling everything to fit a specific values range or something like that. 

This block has two parts:
 - plain math that finds the "center/middle" of the pattern (called "mean")
 - plain math that finds the average length of pattern rays from that center
 - shifting the middle of the whole pattern to zero spot on all axes, so it's now "centered"
 - changing pattern proportions by making its values more statistically average in compare to each other, so we don't have any more single too big spikes or single too short values. At that we also compress the range making values smaller.
   <br><br>
 - multiplying each axis value by a number that model has learnt for this given axis (changing ray length)
 - adding to each axis a fixed number that model learnt to apply here for this given axis (changing ray length)

I don't know how to comment upon this, as at this moment i feel like crying. I know this is empirically considered to be a great solution that works, but it makes me only cry more. It all is about methodology, after all.

So, let's first explain why it's used at all. 

We have several tasks here in the *current* arch:
 - to produce some stable and more predictable input for the next block
 - at that to prevent losing small distinctions when compressing the values range 
 (as in a fixed range a single too big axis value can make all other values be so small, that they will share the same value and lose all its distinctions, while with this method the axes don't become equal if they were not originally)
 
The thing is that the next block (FFN) anyways sees *only* this centered and averaged representation, so it believes this is the ground truth and it learns to extract data from this thing. But it doesn't mean there was no data loss at normalization or that it's not a bottleneck. 

#### start of critique
Let's concentrate on the fact that we introduce two learnt parameters per axis: a value to multiplicate the ray by and a value to add to the ray's length. These are a two learnt values per axis, and they do the same to *any* token produced by attention with *any* figures inside. 

Now let's see the problems it creates. At this point we already break: 
 - absolute number ranges
 - the actual figures we had in the pattern
 - the skew of these figures
 
The actual *figures* can still be used, however *only within of a limited range of values* as if some elephant trunk is too long, it gets averaged and feels then like elephant has silently visited a plastic surgery clinic to become a k-idol. Even if model tried to encode some relatedness via trunk length during the learning, it failed and had to find another way. Because if our token has no other prominent axes, it would shrink. But if our token would have other high values, it would *not* shrink this much. So the value becomes dependent not just on mingled figures, but its *passage* becomes dependant on what other figures are there. And now it can't vary the length of a *single* axis, as it becomes much less predictable.

Also, as we apply the *same* gain+shift per axis to *every different mingled token*, the model *can not* rely on the absolute values ranges from now on as this method of passing the information becomes almost impredictable. All of a sudden the range of already established pattern can change once average length of all rays has changed.

The skewing of figures can not be used, because normalization introduces *uneven* distortion as border values will change more, while the average values will remain somewhat the same. It partially applies to encoding with figures as well.

All of this, as i reason here, should enforce model to rely upon figures consisting of *average* similar values, and limits its methods to encode information. 

So now, model has to encode relatedness through proportions of specific groups of axes, rather than through all of them, because changing a single axis may change proportions and values of *other* axes. So the most reliable way to ensure the signal passes, is to isolate certain group of axes by some specific gain, so that this specific combination always passes the normalization gate with the same distortion level. The model is enforced to tie certain types of figures to certain axes as they have specific gain+shift values it finds during training and these are *compatible*, while some other axes might be compatible with other figures.

Thus, our normalization layer is a huge noise source, significantly complicating for the model the means of delivering its signal to the next block. And it's also a huge bottleneck as it limits the ways for model to encode the meaning.

It's like telling the model: "Wait, we have screwed your patterns by compressing and averaging them, each one in a different way, so now try to find how to change every of the rays to fix it. And sorry, you can use only 1 combination for any of the tokens you create".

And the model is like: "!@#EY!821, Umm.. okay.. maybe i can at least try to use this to make groups of axes by giving them common boosts fitting some of my figures, then it's easier to pass through this noisy gate.. Let me then try to waste lots of compute trying to find what at all passes through this pinhole.."

I think that all of this could be avoided if we initially have *stabilized* the signal channel the model uses.

> It could be cool to make V matrix learn on the already mingled tokens but it would require a lot of changes to the kv cache and it's a different story.

Let's work within attention heads with *proportions* directly. Define a range for proportions, mingle tokens by *proportions* first, train V matrix upon the proportions instead of absolute values. Whatever V extracts encode it then via proportions in the same range again. 

This way we *abstract* away from the actual absolute values, tying our model to the proportions channel. And if you believe it limits the means of communication, it's not, as normalization makes absolute values impossible to interpret anyways, it still has to rely upon figures, just it has to find a way to *pass* these through that distortion. 

This way we do *not* need the pattern centering (mean) and compressing/averaging the proportions (standard deviation). We can just pass already stable uniform representation.

This way model can *rely* upon the proportions and their skewing as it becomes a stable channel of information to the next block without all this terrible noise.

This way tokens *keep* not just their internal pattern coherence but they keep their *relative* coherence as we haven't destroyed their size on *per token* average basis. They originally exist in the same range and we only change proportions of the patterns within of that range.

Yes, if we implement proportions lineary, some axis can get oversaturated and clip. It may break the way training works. To avoid this, we can make *non-linear* proportions compression that will normally operate lineary over certain most common wide range, but still allow going *beyond* that range too. Say after "100" we apply a much harder compression, then after "1000" even harder and so on, making clipping nearly impossible.

But this is only half of the fun, as here we also keep absolute numbers in proportion values stable between the blocks too! The model now can encode data not just in figures as in axes proportions, but also in the actual size of the figures as these are now stable and FFN can use these to extract information.

I believe all this should be faster than finding *one size fits all* parameters for normalization and finding the safe values that would pass the normalization, preserving the information.

And it's much cleaner representation for the next block as it is minus one noise source.
#### end of critique

And all of this is called Multiple Heads Attention (MHA). 

Just one more little thing to mention. Transformers have two phases of processing data, one is "prompt evaluation" where the existing text is converted into internal representation and another one is inference, where the new tokens are generated. 
 
So if at the prompt evaluation the target of traits amalgamation is every given token, we just enrich every token with traits from all related preceeding tokens. At the inference we continue that by making a non-existing new token a target for the sum of the whole context.

And upon this, i think we have finished the attention block and can finally move to..

### FFN part.

You may have heard that FFN block stores the *knowledge* but.. all of this is not really accurate, if not to say not even true :). 

To understand the role of FFN you need to know only one thing. On the input FFN receives the tokens table from the multi head attention.. and on the output it produces a pattern that is added up to the very same tokens table it got on the input. Which means.. its output should be compatible with its input :).

So, as you see, FFN just once again "updates" the existing token patterns by doing something on the inside. 

Let's see what it does.

FFN is made of 3 parts:
 - first matrix - extracting data 
 - passing gate and bias
 - second matrix - converting data back

First, let's take a look at what happens:
1. Our MHA resulting token pattern is multiplied with the first FFN matrix which has way more pattern axes
2. Static unique per axis numeric value (bias) is added to the result (per axis)
3. Passing gate (ReLU or sth else) passes only the *new* axes that are actually related to the input
4. Filtered new axes are multiplied into the second matrix, translating it into a different representation pattern with a standard amount of axes (as on input to FFN)
5. A second static unique per axis numeric value (second bias) is added to the result (per axis)
6. Resulting pattern is then mingled with the original tokens table coming from MHA. It's a residual connection, just like after the attention. So the final result is actually a mix of the MHA view with a tuning from FFN.

So, conceptually, this operation finds the relevant figures/patterns stored in FFN and then converts these back into the figures compatible with what we got from MHA, and mingles these in. Just like earlier we were mingling different tokens, here we mingle our tokens with FFN figures reflecting.. what? Token clouds? 

Nope. FFN does not really store some separate standard clouds of tokens. This is because it has to produce only an *adjustment* to the input pattern, not a *new* pattern. Remember, we mingle its result back into original input, so it should learn to *adjust*, not to *translate*.

So what FFN stores is patterns on *how to change* a typical input to shift it closer to some standard pattern. In other words, it finds most probable *adjustments* required to be infused into the MHA pattern, to make it closer to some common token clouds. FFN learns to adjust token clouds, to brush them up, it does not learn separate "standard" token clouds language. 

And if you wondered why do we need the O matrix at all, couldn't we just feed the MHA results to the FFN directly, that's precisely the answer. FFN needs to have the original token representation with MHA changes, to brush it up. It can't work off the pure MHA changes, as they don't have information about the actual related token clouds, it has information only about changes to these clouds. And the meaning of these depends on the *original* token figures. In order to mingle MHA into the original token we have first to translate that faceted representation into a single one. That's why we need one more intermediate step of the O matrix earlier (apart from noise cancellation).

#### How do we mingle MHA into the first matrix 
       ..or "i've finally decided to explain what matrix multiplication is" :). 
       
 - every operation is split by token, so we do all these things per single token of the context - per row from MHA
 - each token coming from MHA is a row of, say, 512 columns/axes
 - FFN's first matrix in multiplication has just as many columns/axes/dimensions as MHA row has - 512 in our example
 - but at that it has way more rows, like x4 times: 2048
 - we do *not* multiply the FFN matrix "into" the MHA row, we do the MHA row *into* each row of FFN matrix by multiplying every column of MHA into the same respective column of FFN. And we do it with the *every row* of the FFN. So the same one input token row multiplies *every* row of the FFN matrix.
 - then we sum up numbers of every of 2048 FFN rows and get 2048 new values/axes/dimensions
 - so we have turned 512 axes into 2048 axes and got a new x4 bigger pattern 
 - we do it for every token we have, returned by MHA

Basically, we did 2048 dot-product comparisons, finding how similar each of our tokens to each of these 2048 patterns.

> this is usually called "matmul" on slang or "matrice multiplication" and that's what we mostly use as a method to extract some information from a pattern, or to "translate" a pattern, which in a way is the same. A lot of people also say "project into another space" which is even more confusing, as they don't even mention if they use an lcd projector or a dlp one. humor.

But what does that process *actually* mean? Well, it's pretty obvious. 

To understand it, we just have to remember what our MHA returned tokens do represent. They represent a pattern configuration, used for expressing their relatedness to other tokens.

When we mingle these into FFN matrix, we simply complete every of 2048 FFN's unfinished patterns, where each of these is a "test" for:
 - "how well"
 - this token matches
 - certain specific trait part
 - of the *most probable adjustments*
 - that FFN's first matrix has learnt during the training
 - to the traits of tokens relatedness
 
How does it work to find these required adjustments?
You can think of these as locks and keys or like some fun test in a magazine where you answer multiple questions.

Our MHA produced relatedness pattern just "unlocks" some of the FFN per axis "locks", by being compared to each of these new axes - these books/locks. If you look at it per axis, it's just the way we compared shapes in Q/K - dot product comparison, just here we do it against 2048 different test patterns. And voila, each new axis/book gives us some value showing how much our MHA pattern relates to it. 

The incoming token here works more as an original noisy signal coming through a set of *detectors* that read it by 2048 own parameters and based on that build a new its own profile of that signal. It's not even quite amplification or dampening, because in the end we get *different* axes/figures. It's not a simple "upscaling" of the same pattern. 

It's like getting results of some psychology test with 512 parameters and comparing it to some totally different psychological test having 2048 different parameters, finding which ones and how much do match. 

Our results here are parameters of that new psychological test, in a totally different parameters representation. A different pattern that encodes the required changes to the original pattern.

As it holds the refinement to the original "profile", by doing it it can actually shift the "profile" to a different final diagnosis :). That is to amend traits so that they move from certain most probable cloud of traits to other related clouds of traits.

#### But.. how does it decide if the new axis is related to the original pattern?
In a classic implementation, the model just learns a simple fixed value that signifies the edge to cross, to be considered relevant. If the result reaches this edge value, then it means "q/k" of the ffn says "yes", this new *axis* is relevant, let's use it in a new pattern. If it's not, we just do not use this axis from FFN at all and pass a zero.

This fixed value (can be negative or positive) is used to detect the edge and is called "bias". Of course it's just a learnt number during the training. 

Basically, we just add it (e.g. plus -2.5 or +4.1, etc) to a FFN axis's value and it just "negates" the typical noise value of the learnt pattern on this axis. 

Then we can look if the sum is now more than zero or not. If the result here surpasses that noise volume level, it means our input data had its say in this. If it's less than or equals zero, we believe all we have is just the standard pattern level "noise" we got after multiplication and this trait is not related.

Please don't get me wrong, we do not compare the result to the bias value itself, we *add* this bias value to the result and then check the sum of this operation with some fixed non-learnt function (RelU, etc). In the original implementation it just checks if it's more than zero or not, as i explained above.

But conceptually, that's just what bias does: it just serves to negate the standard pattern noise level so we could know if our own data is relevant to this axis.

A funny thing here is that bias may be so high that it can allow enriching *any* input patterns with some shift typical for a specific token cloud, to a specific "idea".

Or it can be so low that it will make shifting to certain "ideas" be nearly impossible, unless it's some very rare case.

But in practice as patterns rely upon multiple axes, this is not very likely. It gets balanced in training.

It also mixes two functions into a single fixed learnt value: 

 a) deciding on the axis relatedness by negating the standard learnt value - one size fits all
 
 b) signal amplification/damping that distorts the pattern as it is. 

But as we don't directly infuse that result to the original input and it's a fixed change that is always applied, the next matrix should learn to adjust to it and this distortion shouldn't go further. 

So, we have added the bias value to the comparison value per axis, now what? How do we do know if the axis is related? If the lock was unlocked? How to know if there is any relevance between the MHA learnt cloud of tokens *to the traits* that mark FFN learnt most probable *adjustments* to shift the pattern closer to certain token clouds? *sigh*

> ..the irony of this architecture, is that to measure if there is some useful trait/signal in the incoming token, we first *multiply it* and only *then* check if we had to do the expensive multiplication or not :)

And that's exactly what our second block does - ReLU. It tests every of the new 2048 axes with the added biases to see if the value they have now is positive. If it's more than zero, it considers this axis/trait as a related one and keeps its value on this axis. If the reuslt is negative or zero, it *nullifies* the axis value - our token is not related to the trait of this FFN learnt adjustment. In this case comparison result is just discarded in this axis.

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
