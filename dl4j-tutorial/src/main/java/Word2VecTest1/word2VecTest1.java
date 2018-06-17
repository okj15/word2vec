package Word2VecTest1;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import org.atilika.kuromoji.Token;
import org.atilika.kuromoji.Tokenizer;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class word2VecTest1 {

    public static void main(String[] args) throws IOException {
        // TODO 自動生成されたメソッド・スタブ
        System.out.println(System.getProperty("file.encoding"));

        System.out.println("Load Model...");
        File wordFile = new File("input/model_neologd.vec");
        WordVectors vec = WordVectorSerializer.readWord2VecModel(wordFile);

	String text1 = "1つ目の文章";
	String text2 = "2つ目の文章";
	INDArray vec1 =convertVector(vec, text1);
	INDArray vec2 =convertVector(vec, text2);
	System.out.println(calcCos(vecTarget, vecTmp));
	
    }

    public static INDArray convertVector(WordVectors vec, String text){
        Tokenizer tokenizer = Tokenizer.builder().build();
        List<Token> tokens = tokenizer.tokenize(text);
        ArrayList<String> wordList = new ArrayList<>();
 
        for (Token token : tokens) {
            wordList.add(token.getSurfaceForm());
        }
        Collection<String>  wordAll = vec.vocab().words();
        int count = 0;
        INDArray vecAdd = Nd4j.zeros(1, 300);
        for(String word: wordList){
            if(wordAll.contains(word) && word.length() > 1) {
                vecAdd = vecAdd.add(vec.getWordVectorMatrix(word));
                count++;
            }
        }
        INDArray vecAve = vecAdd.div(count);
        return vecAve;
    }

    public static double calcCos(INDArray vec1, INDArray vec2){

        double cosNum = 0;

        double vec1Length = 0.;
        double vec2Lengrh = 0.;
        double dotProduct = 0.;
        for(int i=0; i<vec1.size(1); i++){
            vec1Length += vec1.getDouble(i) * vec1.getDouble(i);
            vec2Lengrh += vec2.getDouble(i) * vec2.getDouble(i);
            dotProduct += vec1.getDouble(i) * vec2.getDouble(i);
        }

        vec1Length = Math.sqrt(Math.abs(vec1Length));
        vec2Lengrh = Math.sqrt(Math.abs(vec2Lengrh));

        cosNum = dotProduct / (vec1Length*vec2Lengrh);

        return cosNum;
    }
}