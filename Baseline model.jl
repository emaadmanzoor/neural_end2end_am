
ENV["COLUMNS"] = 100
using Pkg; for p in ("Knet","Plots","DelimitedFiles", "Statistics", "Printf", "Random","DataStructures"); haskey(Pkg.installed(),p) || Pkg.add(p); end
using Base.Iterators: flatten
using Knet,DelimitedFiles, Statistics, Printf, Random,DataStructures

#orderedDict is a Datatype from DataStructures.jl that maintains the insertion order
#word,labels alphabet dictionary
words_alphabet = OrderedDict{String,Int}()
labels_alphabet = OrderedDict{String,Int}()
label_index = 1
word_index = 1

function read_conll(pathtofile)
#     word_sentences = []
#     label_sentences = []
    
    global words_alphabet
    global labels_alphabet 
    global label_index
    global word_index 

    word_index_sentences = []
    label_index_sentences = []

    words = []
    labels = []

    word_ids = Int16[]
    label_ids = Int16[]
        
    #parsing CoNLL formatted data
    open(pathtofile,"r") do file
        for ln in eachline(file)
            if length(ln) == 0   
                if length(word_ids) != 0 && length(label_ids) != 0
                    push!(word_index_sentences,word_ids)
                    push!(label_index_sentences,label_ids)
                end
                word_ids = Int16[]
                label_ids = Int16[]    
            else
                tokens = split(ln)
                word_id = 0
                label_id = 0
                
                #-1 is a fallback value, if the word is not in the vocabulary we need to add it otherwise return index
                if get(words_alphabet,String(tokens[2]),-1) == -1
                    words_alphabet[String(tokens[2])] = word_index
                    word_id = word_index
                    word_index += 1 
                else
                    word_id = Int(get(words_alphabet,String(tokens[2]),-1))
                end
                
                
                
                if get(labels_alphabet,String(tokens[5]),-1) == -1
                    labels_alphabet[String(tokens[5])] = label_index
                    label_id = label_index
                    label_index += 1 
                else
                    label_id = Int(get(labels_alphabet,String(tokens[5]),-1))
                end
                
                push!(word_ids,word_id)
                push!(label_ids,label_id)
            end  
        end
    end
    word_index_sentences, label_index_sentences 
end
    

#reading test dataset 
tst_word_index_sentences, tst_label_index_sentences =
                            read_conll("""/Users/moutasem/Desktop/Spring2019/comp541/acl2017-neural_end2end_am-master/data/conll/Paragraph_Level/test.dat""")
println.(summary.((tst_word_index_sentences, tst_label_index_sentences))) 


#reading training dataset
trn_word_index_sentences, trn_label_index_sentences =
                            read_conll("""/Users/moutasem/Desktop/Spring2019/comp541/acl2017-neural_end2end_am-master/data/conll/Paragraph_Level/train.dat""")
println.(summary.((trn_word_index_sentences, trn_label_index_sentences)))   



#reading dev dataset
dev_word_index_sentences, dev_label_index_sentences =
                            read_conll("""/Users/moutasem/Desktop/Spring2019/comp541/acl2017-neural_end2end_am-master/data/conll/Paragraph_Level/dev.dat""")
println.(summary.((dev_word_index_sentences, dev_label_index_sentences)))         



entire_vocabs_array = collect(keys(words_alphabet));
entire_labels_array = collect(keys(labels_alphabet));
summary.((entire_vocabs_array,entire_labels_array))

struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)
(c::Chain)(x,y) = nll(c(x),y)

struct Dense; w; b; f; end
Dense(i::Int,o::Int,f=identity) = Dense(param(o,i), param0(o), f)
(d::Dense)(x) = d.f.(d.w * mat(x,dims=1) .+ d.b)

struct Embed; w; end
Embed(vocabsize::Int,embedsize::Int) = Embed(param(embedsize,vocabsize))
(e::Embed)(x) = e.w[:,x]

Tagger(vocab,embed,hidden,output)=  # MLP Tagger
    Chain(Embed(vocab,embed),Dense(embed,hidden,relu),Dense(hidden,output))

BATCHSIZE = 32
SEQLENGTH = 16;

function seqbatch(x,y,B,T)
    N = length(x) ÷ B
    x = permutedims(reshape(x[1:N*B],N,B))
    y = permutedims(reshape(y[1:N*B],N,B))
    d = []
    for i in 0:T:N-T
        push!(d, (x[:,i+1:i+T], y[:,i+1:i+T]))
    end
    return d
end

allw_trn = vcat(trn_word_index_sentences...)
allt_trn = vcat(trn_label_index_sentences...)
allw_tst = vcat(tst_word_index_sentences...)
allt_tst = vcat(tst_label_index_sentences...)
dtrn = seqbatch(allw_trn, allt_trn, BATCHSIZE, SEQLENGTH);
dtst = seqbatch(allw_tst, allt_tst, BATCHSIZE, SEQLENGTH);

VOCABSIZE = length(entire_vocabs_array)
EMBEDSIZE = 128
HIDDENSIZE = 128
OUTPUTSIZE = length(entire_labels_array);

# shuffle and split minibatches into train and test portions
shuffle!(dtrn)
shuffle!(dtst)
length.((dtrn,dtst))

function trainresults(file,model,savemodel)
    if (print("Train from scratch? "); readline()[1]=='y')
        takeevery(n,itr) = (x for (i,x) in enumerate(itr) if i % n == 1)
        results = ((nll(model,dtst), zeroone(model,dtst))
                   for x in takeevery(100, progress(adam(model,repeat(dtrn,5)))))
        results = reshape(collect(Float32,flatten(results)),(2,:))
        Knet.save(file,"model",(savemodel ? model : nothing),"results",results)
        Knet.gc() # To save gpu memory
    else
        isfile(file) || download("http://people.csail.mit.edu/deniz/models/tutorial/$file",file)
        model,results = Knet.load(file,"model","results")
    end
    println(minimum(results,dims=2))
    return model,results
end

tagger = Tagger(VOCABSIZE,EMBEDSIZE,HIDDENSIZE,OUTPUTSIZE)


(t,results) = trainresults("tagger113a.jld2",tagger,false);


using Plots; default(fmt=:png,ls=:auto,ymirror=true)

plot([results[2,:]]; xlabel="x100 updates", ylabel="error",
    ylim=(0,5), yticks=0:1:10, labels=["MLP"])


