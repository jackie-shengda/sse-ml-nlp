package ml;

import ml.text.functions.TextPipeline;
import nlp.tokenizations.tokenizerFactory.ChineseTokenizerFactory;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.EmbeddingLayer;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by Jackie.S on 2017/7/25.
 */
public class trainner {


    private int VOCAB_SIZE = 512;
    private JavaSparkContext jsc;

    public void tainning() throws Exception {
        MultiLayerConfiguration netconf = new NeuralNetConfiguration.Builder()
                .seed(1234)
                .iterations(1)
                .learningRate(0.1)
                .learningRateScoreBasedDecayRate(0.5)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .regularization(true)
                .l2(5 * 1e-4)
                .updater(Updater.ADAM)
                .list()
                .layer(0, new EmbeddingLayer.Builder().nIn(VOCAB_SIZE).nOut(512).activation("identity").build())
                .layer(1, new GravesLSTM.Builder().nIn(512).nOut(512).activation("softsign").build())
                .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax").nIn(512).nOut(2).build())
                .pretrain(false).backprop(true)
                .setInputType(InputType.recurrent(VOCAB_SIZE))
                .build();

        Map<String, Object> TokenizerVarMap = new HashMap<>();      //定义文本处理的各种属性
        TokenizerVarMap.put("numWords", 1);     //词最小出现次数
        TokenizerVarMap.put("nGrams", 1);       //language model parameter
        TokenizerVarMap.put("tokenizer", ChineseTokenizerFactory.class.getName());  //分词器实现
        TokenizerVarMap.put("tokenPreprocessor", CommonPreprocessor.class.getName());
        TokenizerVarMap.put("useUnk", true);    //unlisted words will use usrUnk
        TokenizerVarMap.put("vectorsConfiguration", new VectorsConfiguration());
        TokenizerVarMap.put("stopWords", new ArrayList<String>());  //stop words
        Broadcast<Map<String, Object>>  broadcasTokenizerVarMap = jsc.broadcast(TokenizerVarMap);   //broadcast the parameter map


        JavaRDD<String> javaRDDCorpus = jsc.textFile("./src/main/java/resources/corpus.txt");
        TextPipeline textPipeLineCorpus = new TextPipeline(javaRDDCorpus, broadcasTokenizerVarMap);
        JavaRDD<List<String>> javaRDDCorpusToken = textPipeLineCorpus.tokenize();   //tokenize the corpus
        textPipeLineCorpus.buildVocabCache();                                       //build and get the vocabulary
        textPipeLineCorpus.buildVocabWordListRDD();                                 //build corpus
        Broadcast<VocabCache<VocabWord>> vocabCorpus = textPipeLineCorpus.getBroadCastVocabCache();
        JavaRDD<List<VocabWord>> javaRDDVocabCorpus = textPipeLineCorpus.getVocabWordListRDD(); //get tokenized corpus

       /* JavaRDD<DataSet> javaRDDTrainData = javaPairRDDVocabLabel.map(new Function<Tuple2<List<VocabWord>,VocabWord>, DataSet>() {

            @Override
            public DataSet call(Tuple2<List<VocabWord>, VocabWord> tuple) throws Exception {
                List<VocabWord> listWords = tuple._1;
                VocabWord labelWord = tuple._2;
                INDArray features = Nd4j.create(1, 1, maxCorpusLength);
                INDArray labels = Nd4j.create(1, (int)numLabel, maxCorpusLength);
                INDArray featuresMask = Nd4j.zeros(1, maxCorpusLength);
                INDArray labelsMask = Nd4j.zeros(1, maxCorpusLength);
                int[] origin = new int[3];
                int[] mask = new int[2];
                origin[0] = 0;                        //arr(0) store the index of batch sentence
                mask[0] = 0;
                int j = 0;
                for (VocabWord vw : listWords) {         //traverse the list which store an entire sentence
                    origin[2] = j;
                    features.putScalar(origin, vw.getIndex());
                    //
                    mask[1] = j;
                    featuresMask.putScalar(mask, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
                    ++j;
                }
                //
                int lastIdx = listWords.size();
                int idx = labelWord.getIndex();
                labels.putScalar(new int[]{0,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
                labelsMask.putScalar(new int[]{0,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
                return new DataSet(features, labels, featuresMask, labelsMask);
            }
        });
*/
    }
}
