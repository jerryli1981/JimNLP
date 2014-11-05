package edu.pengli.nlp.conference.acl2015.pipe;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Set;
import java.util.TreeSet;
import java.util.Map.Entry;

import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.platform.types.FeatureVector;

public class FeatureVectorGenerator {
	
	private HashMap<String, float[]> wordMap;

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
		String sentence = p.toString();
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
