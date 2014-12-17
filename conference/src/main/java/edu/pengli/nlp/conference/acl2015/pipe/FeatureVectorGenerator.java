package edu.pengli.nlp.conference.acl2015.pipe;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.Map.Entry;

import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLCell;

import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.util.CoreMap;

public class FeatureVectorGenerator {
	
	private HashMap<String, float[]> wordMap;
	
	private Alphabet dictionary;

	int max_size; // max length of strings
	int N; // number of closest words that will be shown
	int max_w; // max length of vocabulary entries
	int dimension;
	
	public FeatureVectorGenerator(){
		wordMap = new HashMap<String, float[]>();
		max_size = 2000; 
		N = 40; 
		max_w = 50; 
        String modelPath = "/home/peng/Develop/Workspace/Mavericks/models"
        		+ "/word2vec/GoogleNews-vectors-negative300.bin";
		try {
			loadModel(modelPath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
		
	private ArrayList<MLDouble> generateMatlabInput
	(InstanceList instances, String name, int maxPatternSize){	
		ArrayList<int[]> matrix = new ArrayList<int[]>();
		ArrayList<int[]> lbl_matrix = new ArrayList<int[]>();
		for(Instance inst : instances){
			ArrayList<IndexedWord> toks = (ArrayList<IndexedWord>)inst.getData();
			int[] idx_arr = new int[maxPatternSize+1];
			int[] lbl_arr = new int[2];
			for(int i=0; i<idx_arr.length; i++){
				if(i < toks.size()){
					int idx = dictionary.lookupIndex(toks.get(i).originalText());
					if(idx >= dictionary.size()){
						System.out.println("Impossible of lookup");
						System.exit(0);
					}
					idx_arr[i] = idx+1;
				}else{
					idx_arr[i] = dictionary.size()+1;
				}
			}
			matrix.add(idx_arr);
			String label = (String)inst.getTarget();
			lbl_arr[0] = Integer.parseInt(label);
			lbl_arr[1] = toks.size();
			lbl_matrix.add(lbl_arr);
		}
		double[] arr = new double[(maxPatternSize+1)*instances.size()];
		int c = 0;
		for(int i=0; i<maxPatternSize+1; i++){
			for(int[] ints : matrix)
				arr[c++] = ints[i];						
		}
		
		double[] lbl_arr = new double[2*instances.size()];
		c = 0;
		for(int i=0; i<2; i++){
			for(int[] ints : lbl_matrix)
				lbl_arr[c++] = ints[i];						
		}
	
		ArrayList<MLDouble> ret = new ArrayList<MLDouble>();
		ret.add(new MLDouble(name, arr, instances.size()));
		ret.add(new MLDouble(name+"_lbl", lbl_arr, instances.size()));
		return ret;
	}
	public FeatureVectorGenerator(HashSet<Pattern> patternSet) throws IOException{
		dictionary = new Alphabet();
		int maxPatternSize = 0;
		InstanceList instances = new InstanceList(null);
		HashSet<String> set = new HashSet<String>();
		for(Pattern p : patternSet){
			CoreMap sentence = p.getAnnotatedSentence();
			
			SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			ArrayList<IndexedWord> list = new ArrayList<IndexedWord>();
			StringBuilder sb = new StringBuilder();
			for (int i = 0; i < labels.size() - 1; i++) {
				CoreLabel nextToken = labels.get(i+1);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(nextToken.index());
				if(p.getArg1().contains(word) || p.getRel().contains(word)
						|| p.getArg2().contains(word))
					continue;
				if(word != null){
					list.add(word);
					dictionary.lookupIndex(word.originalText());
					sb.append(word.originalText()+" ");	
				}
								
			}
			
			if(!set.contains(sb.toString().trim())){
				set.add(sb.toString().trim());
				Instance inst = new Instance(list, "2", null);
				instances.add(inst);
			}
			ArrayList<IndexedWord> list2 = new ArrayList<IndexedWord>();
			
			int size = 0;
			for(IndexedWord iw : p.getArg1()){
				size++;
				dictionary.lookupIndex(iw.originalText());
				list2.add(iw);
			}
			for(IndexedWord iw : p.getRel()){
				size++;
				dictionary.lookupIndex(iw.originalText());
				list2.add(iw);
			}
			for(IndexedWord iw : p.getArg2()){
				size++;
				dictionary.lookupIndex(iw.originalText());
				list2.add(iw);
			}		
			
			instances.add(new Instance(list2, "1", null));
			if(size >= maxPatternSize)
				maxPatternSize = size;
		}
		
		InstanceList training = new InstanceList(null);
		InstanceList validating = new InstanceList(null);
		InstanceList testing  = new InstanceList(null);
		Random rand = new Random();
		int size = instances.size();
		int newSize = size;
		for(int i=0; i< size*0.7; i++){
			int ran = rand.nextInt(newSize);
			training.add(instances.get(ran));
			instances.remove(ran);
			newSize--;
		}
		for(int i=0; i< size*0.2; i++){
			int ran = rand.nextInt(newSize);
			validating.add(instances.get(ran));
			instances.remove(ran);
			newSize--;
		}
		testing.addAll(instances);
				
		String[] idx_arr = new String[dictionary.size()];
		int[] dims = new int[dictionary.size()];
		for(int i=0; i<dictionary.size(); i++){
			idx_arr[i] = (String) dictionary.lookupObject(i);
			dims[i] = i+1;
		}
		MLCell cell = new MLCell("index", dims);
		for(int i=0; i<dictionary.size(); i++){
			MLArray val = new MLChar("index", idx_arr[i]);
			cell.set(val, 1, 1);
		}
		
		double[] vocSize_arr = new double[1];
		vocSize_arr[0] = dictionary.size()+1;
		
		double[] sentLength_arr = new double[1];
		sentLength_arr[0] = maxPatternSize+1;
		ArrayList list = new ArrayList();
		list.add(cell);
		list.add(new MLDouble("sent_length", sentLength_arr, 1));
		list.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list.addAll(generateMatlabInput(testing, "test", maxPatternSize));
		list.addAll(generateMatlabInput(training, "train", maxPatternSize));
		list.addAll(generateMatlabInput(validating, "valid", maxPatternSize));

		new MatFileWriter("/home/peng/Develop/Workspace_matlab/DCNN/Data/mat_file.mat", list);
		System.out.println("done");
	}
	
	public int getDimension(){
		return dimension;
	}
	
	private void loadModel(String modelPath) throws IOException{
		
		BufferedInputStream bis = new BufferedInputStream(new FileInputStream(
				modelPath));
		DataInputStream dis = new DataInputStream(bis);
		int words, size;
		double len;
		String firstLine = dis.readLine();
		words = Integer.parseInt(firstLine.split(" ")[0]);
		size = Integer.parseInt(firstLine.split(" ")[1]);
		dimension = size;
		String word;
		float[] vectors = null;
		for (int b = 0; b < words; b++) {
			word = readString(dis);
			vectors = new float[size];
			len = 0;
			for (int a = 0; a < size; a++) {
				float vector = readFloat(dis);
				len += vector * vector;
				vectors[a] = vector;
			}
			len = (float) Math.sqrt(len);
			
			for (int a = 0; a < size; a++) {
				vectors[a] /= len;
			}		
			wordMap.put(word, vectors);
		}
		bis.close();
		dis.close();
		
	}
	
	public FeatureVector getFeatureVector(String[] keywords){
		double[] vec = new double[dimension];
		for(String word : keywords){
			float[] wordVector = wordMap.get(word);
			if(wordVector == null)
				continue;
			for (int a = 0; a < dimension; a++) {
				vec[a] += wordVector[a];
			}
		}
		float len = 0;
		for (int a = 0; a < dimension; a++) {
			len += vec[a] * vec[a];
		}
		len = (float) Math.sqrt(len);
		for (int a = 0; a < dimension; a++) {
			vec[a] /= len;
		}
		
		int[] idx = new int[dimension];
		for(int a = 0; a <dimension; a++){
			idx[a] = a;
		}
		FeatureVector fv = new FeatureVector(idx, vec);
		return fv;
	}
	
	public FeatureVector getFeatureVector(Pattern p){
		String sentence = p.toGeneralizedForm();
		String[] words = sentence.split(" ");
		double[] vec = new double[dimension];
		for(String word : words){
			float[] wordVector = wordMap.get(word);
			if(wordVector == null)
				continue;
			for (int a = 0; a < dimension; a++) {
				vec[a] += wordVector[a];
			}
		}
		float len = 0;
		for (int a = 0; a < dimension; a++) {
			len += vec[a] * vec[a];
		}
		len = (float) Math.sqrt(len);
		for (int a = 0; a < dimension; a++) {
			vec[a] /= len;
		}
		
		int[] idx = new int[dimension];
		for(int a = 0; a <dimension; a++){
			idx[a] = a;
		}
		FeatureVector fv = new FeatureVector(idx, vec);
		return fv;
	}
	
	public Set<WordEntry> distance(String sentence) {
		
		String[] words = sentence.split(" ");
		float[] vec = new float[dimension];
		for(String word : words){
			float[] wordVector = wordMap.get(word);
			if(wordVector == null)
				return null;
			for (int a = 0; a < dimension; a++) {
				vec[a] += wordVector [a];
			}
		}
		
		float len = 0;
		for (int a = 0; a < dimension; a++) {
			len += vec[a] * vec[a];
		}
		len = (float) Math.sqrt(len);
		for (int a = 0; a < dimension; a++) {
			vec[a] /= len;
		}
	
		Set<Entry<String, float[]>> entrySet = wordMap.entrySet();
		float[] tempVector = null;
		List<WordEntry> wordEntrys = new ArrayList<WordEntry>(N);
		String name = null;
		for (Entry<String, float[]> entry : entrySet) {
			name = entry.getKey();
			boolean flag = false;
			for(String word : words){
				if (name.equals(word)) {
					flag = true;
				}
			}
			if(flag == true)
				continue;

			float dist = 0;
			tempVector = entry.getValue();
		
			for (int i = 0; i < dimension; i++) {
				dist += vec[i] * tempVector[i];
			}
			insertTopN(name, dist, wordEntrys);
		}
		
		return new TreeSet<WordEntry>(wordEntrys);
	}
	
	private void insertTopN(String name, double score,
			List<WordEntry> wordsEntrys) {

		if (wordsEntrys.size() < N) {
			wordsEntrys.add(new WordEntry(name, score));
			return;
		}
		double min = Float.MAX_VALUE;
		int minOffe = 0;
		for (int i = 0; i < N; i++) {
			WordEntry wordEntry = wordsEntrys.get(i);
			if (min > wordEntry.score) {
				min = wordEntry.score;
				minOffe = i;
			}
		}

		if (score > min) {
			wordsEntrys.set(minOffe, new WordEntry(name, score));
		}

	}
	
	public class WordEntry implements Comparable<WordEntry> {
		public String name;
		public double score;

		public WordEntry(String name, double score) {
			this.name = name;
			this.score = score;
		}

		@Override
		public String toString() {
			// TODO Auto-generated method stub
			return this.name + "\t" + score;
		}

		@Override
		public int compareTo(WordEntry o) {
			// TODO Auto-generated method stub
			if (this.score > o.score) {
				return -1;
			} else {
				return 1;
			}
		}

	}
	
	public static float readFloat(InputStream is) throws IOException {
		byte[] bytes = new byte[4];
		is.read(bytes);
		return getFloat(bytes);
	}
	
	public static float getFloat(byte[] b) {
		int accum = 0;
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}
	
	private String readString(DataInputStream dis) throws IOException {
		// TODO Auto-generated method stub
		//</s> in for that is on
		byte[] bytes = new byte[max_w];
		byte b = dis.readByte();
		int i = -1;
		StringBuilder sb = new StringBuilder();
		while (b != 32 && b != 10) { //32 space, 10 newline
			i++;
			bytes[i] = b;
			b = dis.readByte();
			if (i == 49) {
				sb.append(new String(bytes));
				i = -1;
				bytes = new byte[max_w];
			}
		}
		sb.append(new String(bytes, 0, i+1));
		return sb.toString();
	}
}
