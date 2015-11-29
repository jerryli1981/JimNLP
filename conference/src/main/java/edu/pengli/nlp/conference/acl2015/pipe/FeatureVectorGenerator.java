package edu.pengli.nlp.conference.acl2015.pipe;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLCell;

import edu.pengli.nlp.conference.acl2015.types.Category;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.pipe.Pipe;
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

public class FeatureVectorGenerator extends Pipe{
	
	
	private HashMap<Instance, FeatureVector> instanceVectorMap = null;
	private HashMap<String, float[]> wordMap = null;
	
	public FeatureVectorGenerator(){
		instanceVectorMap = new HashMap<Instance, FeatureVector>();
//		wordMap = new HashMap<String, float[]>();
//		initializeWordVectorMap();
	}
	
	public HashMap<String, float[]> getWordMap(){
		return wordMap;
	}
	
	
	protected Instance pipe(Instance inst) {
		
		FeatureVector fv = instanceVectorMap.get(inst);
		return new Instance(fv, null, null, inst.getSource());
	}
			
	
	public void trainingPatternDCNN(String outputSummaryDir, InstanceList patternList, 
			MatlabProxy proxy) throws IOException, MatlabInvocationException{
		
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" +"ALL" + ".dict.ser"));
		
		Alphabet dictionary = new Alphabet();
		// put all seed keywords into dictionary
		for(int i=1; i<= 5; i++){
			Map<String, String[]> aspects = Category.getAspects(i);
			Set<String> keys = aspects.keySet();
			for (String key : keys) {
				String[] keywords = aspects.get(key);
				for(String w : keywords){
					w = w.replaceAll(" ", "_");
					dictionary.lookupIndex(w);	
				}
			}
		}
		
		// put all pattern and tuple mentions into dictionary
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
			wordList.addAll(p.getArg1());
			wordList.addAll(p.getRel());
			wordList.addAll(p.getArg2());
			for(IndexedWord iw : wordList){
				String wordMention = iw.originalText();
				wordMention = wordMention.replaceAll(" ", "_");
				dictionary.lookupIndex(wordMention);	
			}
			String arg1Type = p.getArg1().getHead().ner().toUpperCase();
			arg1Type = arg1Type.replaceAll(" ", "_");
			dictionary.lookupIndex(arg1Type);
			
			String arg2Type = p.getArg2().getHead().ner().toUpperCase();
			arg2Type = arg2Type.replaceAll(" ", "_");
			dictionary.lookupIndex(arg2Type);
			
		}
		
		int maxPatternSize = 0;
		InstanceList instances = new InstanceList(null);
		HashSet<String> set = new HashSet<String>();
		//also need put all other tuples into dictionary
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			CoreMap sentence = p.getTuple().getAnnotatedSentence();
			SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			StringBuilder negativePattern = new StringBuilder();
			for (int i = 0; i < labels.size(); i++) {
				CoreLabel token = labels.get(i);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(token.index());
				if(p.getArg1().contains(word) || p.getRel().contains(word)
						|| p.getArg2().contains(word))
					continue;
								
				if(word != null){
					String wordMention = word.originalText();
					wordMention = wordMention.replaceAll(" ", "_");
					negativePattern.append(wordMention+" ");
					dictionary.lookupIndex(wordMention);
				}else{
					String tokenMention = token.originalText();
					tokenMention = tokenMention.replaceAll(" ", "_");
					negativePattern.append(tokenMention+" ");
					dictionary.lookupIndex(tokenMention);
				}
			}

		
			//pattern cover original sentence, so there is no negative pattern					
			if(!negativePattern.toString().trim().equals("") && 
					!set.contains(negativePattern.toString().trim())){
				set.add(negativePattern.toString().trim());
				Instance negativeInst = new Instance(negativePattern.toString().trim(), 
						"2", "negativePattern", sentence);
				instances.add(negativeInst);
			}
			
			StringBuilder positivePattern = new StringBuilder();

			int size = 0;
			String arg1Type = p.getArg1().getHead().ner().toUpperCase();
			arg1Type = arg1Type.replaceAll(" ", "_");
			for(IndexedWord iw : p.getArg1()){
				size++;
				if(iw.index() == p.getArg1().getHead().index()){
					positivePattern.append(arg1Type+" ");
				}else
					positivePattern.append(iw.originalText().replaceAll(" ", "_")+" ");
			}
				
			for(IndexedWord iw : p.getRel()){
				size++;
				String mention = iw.originalText();
				mention = mention.replaceAll(" ", "_");
				positivePattern.append(mention+" ");
			}

			String arg2Type = p.getArg2().getHead().ner().toUpperCase();
			arg2Type = arg2Type.replaceAll(" ", "_");
			for(IndexedWord iw : p.getArg2()){
				size++;	
				if(iw.index() == p.getArg2().getHead().index()){
					positivePattern.append(arg2Type+" ");
				}else
					positivePattern.append(iw.originalText().replaceAll(" ", "_")+" ");
			}		
				
			Instance positiveInst = new Instance(positivePattern.toString().trim(), 
					"1", "positivePattern", sentence);

			instances.add(positiveInst);
			if(size >= maxPatternSize)
				maxPatternSize = size;
		}
		
		ArrayList<String[]> training = new ArrayList<String[]>();
		ArrayList<String[]> validating = new ArrayList<String[]>();
		ArrayList<String[]> testing  = new ArrayList<String[]>();
	
		Random rand = new Random();
		int size = instances.size();
		int newSize = size;
		for(int i=0; i< size*0.7; i++){
			int ran = rand.nextInt(newSize);
			String instLabel = (String)instances.get(ran).getTarget();
			String[] map = new String[2];
			map[0] = (String)instances.get(ran).getData();
			map[1] = instLabel;
			training.add(map);
			instances.remove(ran);
			newSize--;
		}
		for(int i=0; i< size*0.2; i++){
			int ran = rand.nextInt(newSize);
			String instLabel = (String)instances.get(ran).getTarget();
			String[] map = new String[2];
			map[0] = (String)instances.get(ran).getData();
			map[1] = instLabel;
			validating.add(map);
			instances.remove(ran);
			newSize--;
		}
		for(Instance inst : instances){
			String instLabel = (String)inst.getTarget();
			String[] map = new String[2];
			map[0] = (String)inst.getData();
			map[1] = instLabel;
			testing.add(map);
		}					
		int[] dims = new int[2];
		dims[0] = dictionary.size();
		dims[1] = 1;
		MLCell cell = new MLCell("index", dims);
		for(int i=0; i<dictionary.size(); i++){
			String value = (String)dictionary.lookupObject(i);
			MLArray val = new MLChar("string", value);
			cell.set(val, i, 0);
		}
		
		double[] vocSize_arr = new double[1];
		vocSize_arr[0] = dictionary.size()+1;
		
		double[] sentLength_arr = new double[1];
		sentLength_arr[0] = maxPatternSize+1;
		ArrayList list = new ArrayList();
		list.add(cell);

		list.add(new MLDouble("sent_length", sentLength_arr, 1));
		list.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list.addAll(generateMatlabInput(testing, "test", maxPatternSize, dictionary));
		list.addAll(generateMatlabInput(training, "train", maxPatternSize, dictionary));
		list.addAll(generateMatlabInput(validating, "valid", maxPatternSize, dictionary));

		int sizeOfWordVector = 50;
		double[] vec = new double[sizeOfWordVector*dictionary.size()];
		int c = 0;
		
		for(int i=0; i<dictionary.size(); i++){
			String word = (String)dictionary.lookupObject(i);
			float[] wordVector = wordMap.get(word);
			if(wordVector == null){
				for (int a = 0; a < sizeOfWordVector; a++) {
					vec[c++] = rand.nextDouble();
				}
			}else{
				for (int a = 0; a < sizeOfWordVector; a++) {
					vec[c++] = wordVector[a];
				}
			}
		
		}
		list.add(new MLDouble("vocab_emb", vec, sizeOfWordVector));
		String matInputFile = outputSummaryDir + "/" + "ALL" + "_In.mat";
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model.mat";

		new MatFileWriter(matInputFile, list);
		
		out.writeObject(dictionary);
		out.writeInt(maxPatternSize);
		out.close();
		
		proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
				+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
		proxy.eval("Train('"+matInputFile+"', '"+modelOutputFile+"')");
				
	}
	
	public void trainingTupleDCNN(String outputSummaryDir, InstanceList patternList, 
			MatlabProxy proxy) throws IOException, MatlabInvocationException{
		
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" +"ALL" + ".dict.tuple.ser"));
		
		Alphabet dictionary = new Alphabet();
	
		// put all tuple mentions into dictionary
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			Tuple t = p.getTuple();
			ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
			wordList.addAll(t.getArg1());
			wordList.addAll(t.getRel());
			wordList.addAll(t.getArg2());
			for(IndexedWord iw : wordList){
				String wordMention = iw.originalText();
				wordMention = wordMention.replaceAll(" ", "_");
				dictionary.lookupIndex(wordMention);	
			}		
		}
		
		int maxTupleSize = 0;
		InstanceList instances = new InstanceList(null);
		HashSet<String> set = new HashSet<String>();
		//also need put all other tokens into dictionary
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			Tuple t = p.getTuple();
			CoreMap sentence = t.getAnnotatedSentence();
			SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			StringBuilder negativeTuple = new StringBuilder();
			for (int i = 0; i < labels.size(); i++) {
				CoreLabel token = labels.get(i);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(token.index());
/*				if(word != null)
					dictionary.lookupIndex(word.originalText().replaceAll(" ", "_"));
				dictionary.lookupIndex(token.originalText().replaceAll(" ", "_"));*/
				dictionary.lookupIndex("1/2");
				
				if(t.getArg1().contains(word) || t.getRel().contains(word)
						|| t.getArg2().contains(word))
					continue;
								
				if(word != null){
					String wordMention = word.originalText();
					wordMention = wordMention.replaceAll(" ", "_");
					negativeTuple.append(wordMention+" ");
					dictionary.lookupIndex(wordMention);
				}else{
					String tokenMention = token.originalText();
					tokenMention = tokenMention.replaceAll(" ", "_");
					negativeTuple.append(tokenMention+" ");
					dictionary.lookupIndex(tokenMention);
				}
			}

		
			//pattern cover original sentence, so there is no negative pattern					
			if(!negativeTuple.toString().trim().equals("") && 
					!set.contains(negativeTuple.toString().trim())){
				set.add(negativeTuple.toString().trim());
				Instance negativeInst = new Instance(negativeTuple.toString().trim(), 
						"2", "negativeTuple", sentence);
				instances.add(negativeInst);
			}
			
			int size = t.getArg1().size() + t.getRel().size() + t.getArg2().size();

			Instance positiveInst = new Instance(t.getSentenceRepresentation().toString().trim(), 
					"1", "positiveTuple", sentence);

			instances.add(positiveInst);
			if(size >= maxTupleSize)
				maxTupleSize = size;
		}
		
		ArrayList<String[]> training = new ArrayList<String[]>();
		ArrayList<String[]> validating = new ArrayList<String[]>();
		ArrayList<String[]> testing  = new ArrayList<String[]>();
	
		Random rand = new Random();
		int size = instances.size();
		int newSize = size;
		for(int i=0; i< size*0.7; i++){
			int ran = rand.nextInt(newSize);
			String instLabel = (String)instances.get(ran).getTarget();
			String[] map = new String[2];
			map[0] = (String)instances.get(ran).getData();
			map[1] = instLabel;
			training.add(map);
			instances.remove(ran);
			newSize--;
		}
		for(int i=0; i< size*0.2; i++){
			int ran = rand.nextInt(newSize);
			String instLabel = (String)instances.get(ran).getTarget();
			String[] map = new String[2];
			map[0] = (String)instances.get(ran).getData();
			map[1] = instLabel;
			validating.add(map);
			instances.remove(ran);
			newSize--;
		}
		for(Instance inst : instances){
			String instLabel = (String)inst.getTarget();
			String[] map = new String[2];
			map[0] = (String)inst.getData();
			map[1] = instLabel;
			testing.add(map);
		}					
		int[] dims = new int[2];
		dims[0] = dictionary.size();
		dims[1] = 1;
		MLCell cell = new MLCell("index", dims);
		for(int i=0; i<dictionary.size(); i++){
			String value = (String)dictionary.lookupObject(i);
			MLArray val = new MLChar("string", value);
			cell.set(val, i, 0);
		}
		
		double[] vocSize_arr = new double[1];
		vocSize_arr[0] = dictionary.size()+1;
		
		double[] sentLength_arr = new double[1];
		sentLength_arr[0] = maxTupleSize+1;
		ArrayList list = new ArrayList();
		list.add(cell);

		list.add(new MLDouble("sent_length", sentLength_arr, 1));
		list.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list.addAll(generateMatlabInput(testing, "test", maxTupleSize, dictionary));
		list.addAll(generateMatlabInput(training, "train", maxTupleSize, dictionary));
		list.addAll(generateMatlabInput(validating, "valid", maxTupleSize, dictionary));

		int sizeOfWordVector = 50;
		double[] vec = new double[sizeOfWordVector*dictionary.size()];
		int c = 0;
		
		for(int i=0; i<dictionary.size(); i++){
			String word = (String)dictionary.lookupObject(i);
			float[] wordVector = wordMap.get(word);
			if(wordVector == null){
				for (int a = 0; a < sizeOfWordVector; a++) {
					vec[c++] = rand.nextDouble();
				}
			}else{
				for (int a = 0; a < sizeOfWordVector; a++) {
					vec[c++] = wordVector[a];
				}
			}
		
		}
		list.add(new MLDouble("vocab_emb", vec, sizeOfWordVector));
		String matInputFile = outputSummaryDir + "/" + "ALL" + "_In_Tuple.mat";
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model_Tuple.mat";

		new MatFileWriter(matInputFile, list);
		
		out.writeObject(dictionary);
		out.writeInt(maxTupleSize);
		out.close();
		
		proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
				+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
		proxy.eval("Train('"+matInputFile+"', '"+modelOutputFile+"')");
				
	}
	
	public void trainingPatternAndTupleDCNN(String outputSummaryDir, InstanceList patternList, 
			MatlabProxy proxy) throws IOException, MatlabInvocationException{
		
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" +"ALL" + ".dict.pt.ser"));
		
		Alphabet dictionary = new Alphabet();
	
		// put all tuple mentions into dictionary
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			Tuple t = p.getTuple();
			ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
			wordList.addAll(t.getArg1());
			wordList.addAll(t.getRel());
			wordList.addAll(t.getArg2());
			for(IndexedWord iw : wordList){
				String wordMention = iw.originalText();
				wordMention = wordMention.replaceAll(" ", "_");
				dictionary.lookupIndex(wordMention);	
			}		
		}
		
		// put all pattern and tuple mentions into dictionary
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
			wordList.addAll(p.getArg1());
			wordList.addAll(p.getRel());
			wordList.addAll(p.getArg2());
			for(IndexedWord iw : wordList){
				String wordMention = iw.originalText();
				wordMention = wordMention.replaceAll(" ", "_");
				dictionary.lookupIndex(wordMention);	
			}
			String arg1Type = p.getArg1().getHead().ner().toUpperCase();
			arg1Type = arg1Type.replaceAll(" ", "_");
			dictionary.lookupIndex(arg1Type);
			
			String arg2Type = p.getArg2().getHead().ner().toUpperCase();
			arg2Type = arg2Type.replaceAll(" ", "_");
			dictionary.lookupIndex(arg2Type);
			
		}
		
		int maxTupleSize = 0;
		int maxPatternSize = 0;
		InstanceList instances = new InstanceList(null);
		HashSet<String> set = new HashSet<String>();
		//also need put all other tokens into dictionary
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			Tuple t = p.getTuple();
			CoreMap sentence = t.getAnnotatedSentence();
			SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			StringBuilder negativeTuple = new StringBuilder();
			for (int i = 0; i < labels.size(); i++) {
				CoreLabel token = labels.get(i);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(token.index());
/*				if(word != null)
					dictionary.lookupIndex(word.originalText().replaceAll(" ", "_"));
				dictionary.lookupIndex(token.originalText().replaceAll(" ", "_"));*/
				dictionary.lookupIndex("1/2");
				
				if(t.getArg1().contains(word) || t.getRel().contains(word)
						|| t.getArg2().contains(word))
					continue;
								
				if(word != null){
					String wordMention = word.originalText();
					wordMention = wordMention.replaceAll(" ", "_");
					negativeTuple.append(wordMention+" ");
					dictionary.lookupIndex(wordMention);
				}else{
					String tokenMention = token.originalText();
					tokenMention = tokenMention.replaceAll(" ", "_");
					negativeTuple.append(tokenMention+" ");
					dictionary.lookupIndex(tokenMention);
				}
			}

		
			//pattern cover original sentence, so there is no negative pattern					
			if(!negativeTuple.toString().trim().equals("") && 
					!set.contains(negativeTuple.toString().trim())){
				set.add(negativeTuple.toString().trim());
				Instance negativeTupleInst = new Instance(negativeTuple.toString().trim(), 
						"2", "negativeTuple", sentence);
				instances.add(negativeTupleInst);
			}
			
			int tupleSize = t.getArg1().size() + t.getRel().size() + t.getArg2().size();

			Instance positiveTupleInst = new Instance(t.getSentenceRepresentation().toString().trim(), 
					"1", "positiveTuple", sentence);

			instances.add(positiveTupleInst);
			if(tupleSize >= maxTupleSize)
				maxTupleSize = tupleSize;
			
			StringBuilder negativePattern = new StringBuilder();
			for (int i = 0; i < labels.size(); i++) {
				CoreLabel token = labels.get(i);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(token.index());
				if(p.getArg1().contains(word) || p.getRel().contains(word)
						|| p.getArg2().contains(word))
					continue;
								
				if(word != null){
					String wordMention = word.originalText();
					wordMention = wordMention.replaceAll(" ", "_");
					negativePattern.append(wordMention+" ");
					dictionary.lookupIndex(wordMention);
				}else{
					String tokenMention = token.originalText();
					tokenMention = tokenMention.replaceAll(" ", "_");
					negativePattern.append(tokenMention+" ");
					dictionary.lookupIndex(tokenMention);
				}
			}

		
			//pattern cover original sentence, so there is no negative pattern					
			if(!negativePattern.toString().trim().equals("") && 
					!set.contains(negativePattern.toString().trim())){
				set.add(negativePattern.toString().trim());
				Instance negativePatternInst = new Instance(negativePattern.toString().trim(), 
						"2", "negativePattern", sentence);
				instances.add(negativePatternInst);
			}
			
			StringBuilder positivePattern = new StringBuilder();

			int patternSize = 0;
			String arg1Type = p.getArg1().getHead().ner().toUpperCase();
			arg1Type = arg1Type.replaceAll(" ", "_");
			for(IndexedWord iw : p.getArg1()){
				patternSize++;
				if(iw.index() == p.getArg1().getHead().index()){
					positivePattern.append(arg1Type+" ");
				}else
					positivePattern.append(iw.originalText().replaceAll(" ", "_")+" ");
			}
				
			for(IndexedWord iw : p.getRel()){
				patternSize++;
				String mention = iw.originalText();
				mention = mention.replaceAll(" ", "_");
				positivePattern.append(mention+" ");
			}

			String arg2Type = p.getArg2().getHead().ner().toUpperCase();
			arg2Type = arg2Type.replaceAll(" ", "_");
			for(IndexedWord iw : p.getArg2()){
				patternSize++;	
				if(iw.index() == p.getArg2().getHead().index()){
					positivePattern.append(arg2Type+" ");
				}else
					positivePattern.append(iw.originalText().replaceAll(" ", "_")+" ");
			}		
				
			Instance positivePatternInst = new Instance(positivePattern.toString().trim(), 
					"1", "positivePattern", sentence);

			instances.add(positivePatternInst);
			if(patternSize >= maxPatternSize)
				maxPatternSize = patternSize;
			
		}
		
		int maxSize = Math.max(maxTupleSize, maxPatternSize);
		
		ArrayList<String[]> training = new ArrayList<String[]>();
		ArrayList<String[]> validating = new ArrayList<String[]>();
		ArrayList<String[]> testing  = new ArrayList<String[]>();
	
		Random rand = new Random();
		int size = instances.size();
		int newSize = size;
		for(int i=0; i< size*0.7; i++){
			int ran = rand.nextInt(newSize);
			String instLabel = (String)instances.get(ran).getTarget();
			String[] map = new String[2];
			map[0] = (String)instances.get(ran).getData();
			map[1] = instLabel;
			training.add(map);
			instances.remove(ran);
			newSize--;
		}
		for(int i=0; i< size*0.2; i++){
			int ran = rand.nextInt(newSize);
			String instLabel = (String)instances.get(ran).getTarget();
			String[] map = new String[2];
			map[0] = (String)instances.get(ran).getData();
			map[1] = instLabel;
			validating.add(map);
			instances.remove(ran);
			newSize--;
		}
		for(Instance inst : instances){
			String instLabel = (String)inst.getTarget();
			String[] map = new String[2];
			map[0] = (String)inst.getData();
			map[1] = instLabel;
			testing.add(map);
		}					
		int[] dims = new int[2];
		dims[0] = dictionary.size();
		dims[1] = 1;
		MLCell cell = new MLCell("index", dims);
		for(int i=0; i<dictionary.size(); i++){
			String value = (String)dictionary.lookupObject(i);
			MLArray val = new MLChar("string", value);
			cell.set(val, i, 0);
		}
		
		double[] vocSize_arr = new double[1];
		vocSize_arr[0] = dictionary.size()+1;
		
		double[] sentLength_arr = new double[1];
		sentLength_arr[0] = maxTupleSize+1;
		ArrayList list = new ArrayList();
		list.add(cell);

		list.add(new MLDouble("sent_length", sentLength_arr, 1));
		list.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list.addAll(generateMatlabInput(testing, "test", maxSize, dictionary));
		list.addAll(generateMatlabInput(training, "train", maxSize, dictionary));
		list.addAll(generateMatlabInput(validating, "valid", maxSize, dictionary));

		int sizeOfWordVector = 50;
		double[] vec = new double[sizeOfWordVector*dictionary.size()];
		int c = 0;
		
		for(int i=0; i<dictionary.size(); i++){
			String word = (String)dictionary.lookupObject(i);
			float[] wordVector = wordMap.get(word);
			if(wordVector == null){
				for (int a = 0; a < sizeOfWordVector; a++) {
					vec[c++] = rand.nextDouble();
				}
			}else{
				for (int a = 0; a < sizeOfWordVector; a++) {
					vec[c++] = wordVector[a];
				}
			}
		
		}
		list.add(new MLDouble("vocab_emb", vec, sizeOfWordVector));
		String matInputFile = outputSummaryDir + "/" + "ALL" + "_In_PT.mat";
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model_PT.mat";

		new MatFileWriter(matInputFile, list);
		
		out.writeObject(dictionary);
		out.writeInt(maxTupleSize);
		out.close();
		
		proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
				+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
		proxy.eval("Train('"+matInputFile+"', '"+modelOutputFile+"')");
				
	}
	

	
	public void setPatternFvsViaPreTrainedWord2VecModel(String outputSummaryDir, 
			String corpusName, InstanceList patternList){
		ArrayList<FeatureVector> fvs = new ArrayList<FeatureVector>();
		
		//wordEmbeding dimension is 300
		int dimension = 300;
		HashMap<String, float[]> wordMap = null;

		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + corpusName + ".smallWordMap"));
			wordMap = (HashMap<String, float[]>)in.readObject();
			in.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		
		for(Instance inst : patternList){
			Pattern p = (Pattern)inst.getData();
			String Arg1 = p.getArg1().getHead().ner().toLowerCase();
			String Pre = p.getRel().getHead().originalText().toLowerCase();
			String Arg2 = p.getArg2().getHead().ner().toLowerCase();
			float[] wordVectorArg1 = wordMap.get(cleaning(Arg1));
			float[] wordVectorPre = wordMap.get(cleaning(Pre));
			float[] wordVectorArg2 = wordMap.get(cleaning(Arg2));
			if(wordVectorArg1 == null){
				if(Arg1.contains("_")){ 
					String[] toks = Arg1.split("_");
					for(int i=0; i<toks.length; i++){
						wordVectorArg1 = wordMap.get(toks[i]);
					}
				}else if(Arg1.contains(" ")){
					String[] toks = Arg1.split(" ");
					for(int i=0; i<toks.length; i++){
						wordVectorArg1 = wordMap.get(toks[i]);
					}
				}

			}
						
			if(wordVectorArg2 == null){
				if(Arg2.contains("_")){ 
					String[] toks = Arg2.split("_");
					for(int i=0; i<toks.length; i++){
						wordVectorArg2 = wordMap.get(toks[i]);
					}
				}else if(Arg2.contains(" ")){
					String[] toks = Arg2.split(" ");
					for(int i=0; i<toks.length; i++){
						wordVectorArg2 = wordMap.get(toks[i]);
					}
				}
				
			}
						
			double[] vec = new double[dimension*3];
			double[] vec_small = new double[dimension];
			int[] idx = new int[dimension*3];
			int[] idx_small = new int[dimension];
			int c = 0;
			int d = 0;
			for(int i=0; i<wordVectorArg1.length; i++){
				vec[c++] = wordVectorArg1[i];
				idx[d] = d++;
	
			}
			for(int i=0; i<wordVectorPre.length; i++){
				vec[c++] = wordVectorPre[i];
				idx[d] = d++;
			}
			for(int i=0; i<wordVectorArg2.length; i++){
				vec[c++] = wordVectorArg2[i];
				idx[d] = d++;
			}
			for (int a = 0; a < dimension; a++) {
				vec_small[a] += wordVectorArg1[a] + wordVectorPre[a]+
						wordVectorArg2[a];
			}
			
			float len = 0;
			for (int a = 0; a < dimension; a++) {
				len += vec_small[a] * vec_small[a];
			}
			len = (float) Math.sqrt(len);
			for (int a = 0; a < dimension; a++) {
				vec_small[a] /= len;
			}
			
			for(int a = 0; a <dimension; a++){
				idx_small[a] = a;
			}
			
			FeatureVector fv = new FeatureVector(idx, vec);
			fvs.add(fv);	
		}
		
		for(int i=0; i<patternList.size(); i++){
			instanceVectorMap.put(patternList.get(i), fvs.get(i));
		}
	}
	
	public void setSequenceFvsViaPreTrainedWord2VecModel(String outputSummaryDir, 
			String corpusName, InstanceList instanceList){
		
		ArrayList<FeatureVector> fvs = new ArrayList<FeatureVector>();	
		//wordEmbeding dimension is 300
		int dimension = 300;
		HashMap<String, float[]> wordMap = null;

		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + corpusName + ".smallWordMap"));
			wordMap = (HashMap<String, float[]>)in.readObject();
			in.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		for(Instance sequenceInst : instanceList){
			String sequence = (String)sequenceInst.getSource();
			String[] toks= sequence.split(" ");
			double[] vec = new double[dimension];
			int[] idx = new int[dimension];
			for(int a = 0; a <dimension; a++){
				idx[a] = a;
			}
			
			for(String tok : toks){
				tok = cleaning(tok.toLowerCase());
				float[] wordVector = wordMap.get(tok);
				
				if (wordVector == null)
						continue;
				for (int a = 0; a < dimension; a++) {
					vec[a] += wordVector[a];
				}	
				
				float len = 0;
				for (int a = 0; a < dimension; a++) {
					len += vec[a] * vec[a];
				}
				len = (float) Math.sqrt(len);
				for (int a = 0; a < dimension; a++) {
					vec[a] /= len;
				}
			}	
			FeatureVector fv = new FeatureVector(idx, vec);
			fvs.add(fv);	

		}	
		for(int i=0; i<instanceList.size(); i++){
			instanceVectorMap.put(instanceList.get(i), fvs.get(i));
		}
	}
	
	//if use this method, you should train a DCNN pattern model first. 
	public void setPatternFvsViaTrainedDCNN(String outputSummaryDir, String corpusName, InstanceList patternList
			, MatlabProxy proxy){
		Alphabet dictionary = null;
		int maxPatternSize = 0;
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + "ALL" + ".dict.pt.ser"));
			dictionary= (Alphabet)in.readObject();
			maxPatternSize = in.readInt();
			in.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		ArrayList<String> candidates  =new ArrayList<String>();
		// consistent with dictionary construction
		for(Instance inst : patternList){
			StringBuilder positivePattern = new StringBuilder();
			Pattern p = (Pattern)inst.getData();
			String arg1Type = p.getArg1().getHead().ner().toUpperCase();
			arg1Type = arg1Type.replaceAll(" ", "_");
			for(IndexedWord iw : p.getArg1()){
				if(iw.index() == p.getArg1().getHead().index()){
					positivePattern.append(arg1Type+" ");
				}else
					positivePattern.append(iw.originalText().replaceAll(" ", "_")+" ");
			}
				
			for(IndexedWord iw : p.getRel()){
				String mention = iw.originalText();
				mention = mention.replaceAll(" ", "_");
				positivePattern.append(mention+" ");
			}

			String arg2Type = p.getArg2().getHead().ner().toUpperCase();
			arg2Type = arg2Type.replaceAll(" ", "_");
			for(IndexedWord iw : p.getArg2()){
				if(iw.index() == p.getArg2().getHead().index()){
					positivePattern.append(arg2Type+" ");
				}else
					positivePattern.append(iw.originalText().replaceAll(" ", "_")+" ");
			}	
			candidates.add(positivePattern.toString().trim());
		}

		
		ArrayList<String[]> instances  = new ArrayList<String[]>();
		for(String str : candidates){
			String[] inst = new String[2];
			inst[0] = str;
			inst[1] = "1";
			instances.add(inst);
		}
		
		String matInputFile = outputSummaryDir + "/" + "ALL" + "_In_PT.mat";
		MatFileReader red = null;
		try {
			red = new MatFileReader(matInputFile);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		ArrayList list = new ArrayList();
		MLCell cell = (MLCell)red.getMLArray("index");
		list.add(cell);
		MLDouble sent_length = (MLDouble)red.getMLArray("sent_length");
		list.add(sent_length);
		MLDouble size_vocab = (MLDouble)red.getMLArray("size_vocab");
		list.add(size_vocab);
		MLDouble test = (MLDouble)red.getMLArray("test");
		list.add(test);
		MLDouble test_lbl = (MLDouble)red.getMLArray("test_lbl");
		list.add(test_lbl);
		MLDouble train = (MLDouble)red.getMLArray("train");
		list.add(train);
		MLDouble train_lbl = (MLDouble)red.getMLArray("train_lbl");
		list.add(train_lbl);
		
		list.addAll(FeatureVectorGenerator.generateMatlabInput(instances, "valid", maxPatternSize, dictionary));
		
		MLDouble vocab_emb = (MLDouble)red.getMLArray("vocab_emb");
		list.add(vocab_emb);
		
		String matInputFile_AllPosi = outputSummaryDir + "/" + corpusName + "_In_AllPosi.mat";
		try {
			new MatFileWriter(matInputFile_AllPosi, list);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model_PT.mat";
		String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out_AllPosi.mat";
		
		
		ArrayList<FeatureVector> fvs = new ArrayList<FeatureVector>();
		try {
			proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
					+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
			proxy.eval("MyScript('"+modelOutputFile+"',"+"'"+matInputFile_AllPosi+"',"+"'"+matOutputFile+"'"+")");
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		MLDouble data = null;
		try {
			MatFileReader mr = new MatFileReader(matOutputFile);
			data = (MLDouble)mr.getMLArray("M_3");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		double[][] arr = data.getArray();
		int m = data.getM();
		int n = data.getN();
		
		for(int i=0; i<m; i++){
			double[] vec = new double[n];
			int[] idx = new int[n];
			int c = 0;
			for(int j=0; j<n; j++){
				vec[c++] = arr[i][j];
				idx[j] = j;
			}
			FeatureVector fv = new FeatureVector(idx, vec);
			fvs.add(fv);
		}

	
		for(int i=0; i<patternList.size(); i++){
			instanceVectorMap.put(patternList.get(i), fvs.get(i));
		}
		
	}
	
	public void setSequenceFvsViaTrainedDCNN(String outputSummaryDir, 
			String corpusName, InstanceList instanceList
			, MatlabProxy proxy, String flag){
		Alphabet dictionary = null;
		int maxPatternSize = 0;
		try {
			ObjectInputStream in = null;
			if(flag == "pattern"){
				in = new ObjectInputStream(new FileInputStream(
						outputSummaryDir + "/" + "ALL" + ".dict.ser"));
			}else if(flag == "tuple"){
				in = new ObjectInputStream(new FileInputStream(
						outputSummaryDir + "/" + "ALL" + ".dict.tuple.ser"));
			}
			
			dictionary= (Alphabet)in.readObject();
			maxPatternSize = in.readInt();
			in.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		ArrayList<String> candidates  =new ArrayList<String>();
		for(Instance inst : instanceList){
			candidates.add((String)inst.getSource());
		}
	
		ArrayList<String[]> instances  = new ArrayList<String[]>();
		for(String str : candidates){
			String[] inst = new String[2];
			inst[0] = str;
			inst[1] = "1";
			instances.add(inst);
		}
		
		String matInputFile = null;
		
		if(flag == "pattern"){
			matInputFile = outputSummaryDir + "/" + "ALL" + "_In.mat";
		}else if(flag == "tuple"){
			matInputFile = outputSummaryDir + "/" + "ALL" + "_In_Tuple.mat";
		}
		
		MatFileReader red = null;
		try {
			red = new MatFileReader(matInputFile);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		ArrayList list = new ArrayList();
		MLCell cell = (MLCell)red.getMLArray("index");
		list.add(cell);
		MLDouble sent_length = (MLDouble)red.getMLArray("sent_length");
		list.add(sent_length);
		MLDouble size_vocab = (MLDouble)red.getMLArray("size_vocab");
		list.add(size_vocab);
		MLDouble test = (MLDouble)red.getMLArray("test");
		list.add(test);
		MLDouble test_lbl = (MLDouble)red.getMLArray("test_lbl");
		list.add(test_lbl);
		MLDouble train = (MLDouble)red.getMLArray("train");
		list.add(train);
		MLDouble train_lbl = (MLDouble)red.getMLArray("train_lbl");
		list.add(train_lbl);
		
		list.addAll(FeatureVectorGenerator.generateMatlabInput(instances, "valid", maxPatternSize, dictionary));
		
		MLDouble vocab_emb = (MLDouble)red.getMLArray("vocab_emb");
		list.add(vocab_emb);
		
		String matInputFile_AllPosi = outputSummaryDir + "/" + corpusName + "_In_AllPosi_Sequence.mat";
		try {
			new MatFileWriter(matInputFile_AllPosi, list);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		String modelOutputFile = null;
		if(flag == "pattern")
			modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model.mat";
		else if(flag == "tuple")
			modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model_Tuple.mat";
		
		String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out_AllPosi_Sequence.mat";
		
		
		ArrayList<FeatureVector> fvs = new ArrayList<FeatureVector>();
		try {
			proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
					+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
			proxy.eval("MyScript('"+modelOutputFile+"',"+"'"+matInputFile_AllPosi+"',"+"'"+matOutputFile+"'"+")");
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		MLDouble data = null;
		try {
			MatFileReader mr = new MatFileReader(matOutputFile);
			data = (MLDouble)mr.getMLArray("M_3");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		double[][] arr = data.getArray();
		int m = data.getM();
		int n = data.getN();
		
		for(int i=0; i<m; i++){
			double[] vec = new double[n];
			int[] idx = new int[n];
			int c = 0;
			for(int j=0; j<n; j++){
				vec[c++] = arr[i][j];
				idx[j] = j;
			}
			FeatureVector fv = new FeatureVector(idx, vec);
			fvs.add(fv);
		}

	
		for(int i=0; i<instanceList.size(); i++){
			instanceVectorMap.put(instanceList.get(i), fvs.get(i));
		}
		
	}
	
	public void setSequenceFvsViaTrainedDCNN(String outputSummaryDir, 
			String corpusName, InstanceList instanceList
			, MatlabProxy proxy){
		Alphabet dictionary = null;
		int maxPatternSize = 0;
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + "ALL" + ".dict.pt.ser"));
		
			dictionary= (Alphabet)in.readObject();
			maxPatternSize = in.readInt();
			in.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		ArrayList<String> candidates  =new ArrayList<String>();
		for(Instance inst : instanceList){
			candidates.add((String)inst.getSource());
		}
	
		ArrayList<String[]> instances  = new ArrayList<String[]>();
		for(String str : candidates){
			String[] inst = new String[2];
			inst[0] = str;
			inst[1] = "1";
			instances.add(inst);
		}
		
		String matInputFile = outputSummaryDir + "/" + "ALL" + "_In_PT.mat";
			
		MatFileReader red = null;
		try {
			red = new MatFileReader(matInputFile);
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
		ArrayList list = new ArrayList();
		MLCell cell = (MLCell)red.getMLArray("index");
		list.add(cell);
		MLDouble sent_length = (MLDouble)red.getMLArray("sent_length");
		list.add(sent_length);
		MLDouble size_vocab = (MLDouble)red.getMLArray("size_vocab");
		list.add(size_vocab);
		MLDouble test = (MLDouble)red.getMLArray("test");
		list.add(test);
		MLDouble test_lbl = (MLDouble)red.getMLArray("test_lbl");
		list.add(test_lbl);
		MLDouble train = (MLDouble)red.getMLArray("train");
		list.add(train);
		MLDouble train_lbl = (MLDouble)red.getMLArray("train_lbl");
		list.add(train_lbl);
		
		list.addAll(FeatureVectorGenerator.generateMatlabInput(instances, "valid", maxPatternSize, dictionary));
		
		MLDouble vocab_emb = (MLDouble)red.getMLArray("vocab_emb");
		list.add(vocab_emb);
		
		String matInputFile_AllPosi = outputSummaryDir + "/" + corpusName + "_In_AllPosi_Sequence.mat";
		try {
			new MatFileWriter(matInputFile_AllPosi, list);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model_PT.mat";

    	String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out_AllPosi_Sequence.mat";
		
		
		ArrayList<FeatureVector> fvs = new ArrayList<FeatureVector>();
		try {
			proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
					+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
			proxy.eval("MyScript('"+modelOutputFile+"',"+"'"+matInputFile_AllPosi+"',"+"'"+matOutputFile+"'"+")");
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		MLDouble data = null;
		try {
			MatFileReader mr = new MatFileReader(matOutputFile);
			data = (MLDouble)mr.getMLArray("M_3");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		double[][] arr = data.getArray();
		int m = data.getM();
		int n = data.getN();
		
		for(int i=0; i<m; i++){
			double[] vec = new double[n];
			int[] idx = new int[n];
			int c = 0;
			for(int j=0; j<n; j++){
				vec[c++] = arr[i][j];
				idx[j] = j;
			}
			FeatureVector fv = new FeatureVector(idx, vec);
			fvs.add(fv);
		}

	
		for(int i=0; i<instanceList.size(); i++){
			instanceVectorMap.put(instanceList.get(i), fvs.get(i));
		}
		
	}
		
	private static ArrayList<MLDouble> generateMatlabInput
	(ArrayList<String[]> instances, String name, int maxPatternSize, Alphabet dictionary){	
		ArrayList<int[]> matrix = new ArrayList<int[]>();
		ArrayList<int[]> lbl_matrix = new ArrayList<int[]>();
		for(String[] instLabelMap : instances){
			
			String[] toks = instLabelMap[0].split(" ");
			int[] idx_arr = new int[maxPatternSize+1];
			int[] lbl_arr = new int[2];
			for(int i=0; i<idx_arr.length; i++){
				if(i < toks.length){
					int tmpSize = dictionary.size();
					int idx = dictionary.lookupIndex(toks[i]);
					if(idx == tmpSize){
						System.out.println("Impossible of lookup");
						System.exit(0);
					}
					idx_arr[i] = idx+1;
				}else{
					idx_arr[i] = dictionary.size()+1;
				}
			}
			matrix.add(idx_arr);
			String label = instLabelMap[1];
			lbl_arr[0] = Integer.parseInt(label);
			lbl_arr[1] = toks.length;
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

	private void initializeWordVectorMap(){
		System.out.println("Begin to load word vectors");
		
		int max_w = 50; // max length of vocabulary entries
        String modelPath = "/home/peng/Develop/Workspace/Mavericks/models"
        		+ "/word2vec/GoogleNews-vectors-negative300.bin";	

		BufferedInputStream bis;
		try {
			bis = new BufferedInputStream(new FileInputStream(
					modelPath));
	
		DataInputStream dis = new DataInputStream(bis);
		int words, size;
		double len;
		String firstLine = dis.readLine();
		words = Integer.parseInt(firstLine.split(" ")[0]);
		size = Integer.parseInt(firstLine.split(" ")[1]);
		String word;
		float[] vectors = null;
		for (int b = 0; b < words; b++) {
			word = readString(dis, max_w);
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
		
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		System.out.println("load word vectors is done");	
	}
	
	private static float readFloat(InputStream is) throws IOException {
		byte[] bytes = new byte[4];
		is.read(bytes);
		return getFloat(bytes);
	}
	
	private static float getFloat(byte[] b) {
		int accum = 0;
		accum = accum | (b[0] & 0xff) << 0;
		accum = accum | (b[1] & 0xff) << 8;
		accum = accum | (b[2] & 0xff) << 16;
		accum = accum | (b[3] & 0xff) << 24;
		return Float.intBitsToFloat(accum);
	}
	
	private String readString(DataInputStream dis, int max_w) throws IOException {
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
	
	private void generateSmallWordVector(String outputSummaryDir, 
			String corpusName, InstanceList patternList, String categoryId,
			FeatureVectorGenerator fvg){
		//wordEmbeding dimension is 300
		int dimension = 300;
		HashMap<String, float[]> wordMap = fvg.getWordMap();
		HashMap<String, float[]> smallWordMap = new HashMap<String, float[]>();
		
		for(Instance inst : patternList){
			Pattern p = (Pattern)inst.getData();
			Tuple t = p.getTuple();
			ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
			wordList.addAll(t.getArg1());
			wordList.addAll(t.getRel());
			wordList.addAll(t.getArg2());
			for(IndexedWord iw : wordList){
				float[] wv = wordMap.get(iw.originalText());
				smallWordMap.put(iw.originalText(), wv);
			}
			String Arg1 = p.getArg1().getHead().ner().toLowerCase();
			String Pre = p.getRel().getHead().originalText().toLowerCase();
			String Arg2 = p.getArg2().getHead().ner().toLowerCase();
			float[] wordVectorArg1 = wordMap.get(cleaning(Arg1));
			float[] wordVectorPre = wordMap.get(cleaning(Pre));
			float[] wordVectorArg2 = wordMap.get(cleaning(Arg2));
			if(wordVectorArg1 == null){
				if(Arg1.contains("_")){ 
					String[] toks = Arg1.split("_");
					for(int i=0; i<toks.length; i++){
						wordVectorArg1 = wordMap.get(toks[i]);
					}
				}else if(Arg1.contains(" ")){
					String[] toks = Arg1.split(" ");
					for(int i=0; i<toks.length; i++){
						wordVectorArg1 = wordMap.get(toks[i]);
					}
				}

			}
			
			if(wordVectorPre == null){
				System.out.println(cleaning(Pre));
			}
			
			if(wordVectorArg2 == null){
				if(Arg2.contains("_")){ 
					String[] toks = Arg2.split("_");
					for(int i=0; i<toks.length; i++){
						wordVectorArg2 = wordMap.get(toks[i]);
					}
				}else if(Arg2.contains(" ")){
					String[] toks = Arg2.split(" ");
					for(int i=0; i<toks.length; i++){
						wordVectorArg2 = wordMap.get(toks[i]);
					}
				}
				
			}
			
			smallWordMap.put(cleaning(Arg1), wordVectorArg1);
			smallWordMap.put(cleaning(Pre), wordVectorPre);
			smallWordMap.put(cleaning(Arg2), wordVectorArg2);
				
		}
		
		Category[] cats = Category.values();
		for (Category cat : cats) {
			if (cat.getId() == Integer.parseInt(categoryId)) {
				Map<String, String[]> aspects = cat.getAspects(cat.getId());
				Set<String> keys = aspects.keySet();
				for (String k : keys) {
					String[] words = aspects.get(k);
					for (String word : words) {
				
						float[] wordVector = wordMap.get(word);
						if (wordVector == null)
							continue;
						smallWordMap.put(word, wordVector);

					}		
				}
			}
		}	
		
		ObjectOutputStream out;
		try {
			out = new ObjectOutputStream(new FileOutputStream(
					outputSummaryDir + "/" + corpusName + ".smallWordMap"));
			out.writeObject(smallWordMap);
			out.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	private String cleaning(String mention){
		if(mention.equals("'s"))
			return "is";
		else if(mention.equals("cognizer"))
			return "cognize";
		else if(mention.equals("evaluee"))
			return "assess";
		else if(mention.equals("organising") || mention.equals("organised")||mention.equals("organise"))
			return "organize";
		else if(mention.equals("recognise"))
			return "recognize";
		else if(mention.equals("undergoer"))
			return "undergo";
		else if(mention.equals("ingestibles"))
			return "ingest";
		else if(mention.equals("travelled"))
			return "travel";
		else if(mention.equals("abbas-led"))
			return "led";
		else if(mention.equals("ploughed"))
			return "plow";
		else if(mention.equals("internet-based"))
			return "internet";
		else if(mention.equals("criticised"))
			return "criticize";
		else if(mention.equals("uncommunicativeness"))
			return "communicate";
		else if(mention.equals("analysed"))
			return "analyze";
		else if(mention.equals("submittor"))
			return "submit";
		else if(mention.equals("u.s.-led"))
			return "led";
		else if(mention.equals("favours"))
			return "favor";
		else if(mention.equals("stabilise"))
			return "stabilize";
		else if(mention.equals("aggregateproperty"))
			return "aggregate";
		else if(mention.equals("re-establish"))
			return "establish";
		else if(mention.equals("democratic-controlled"))
			return "democratic";
		else if(mention.equals("ill-being"))
			return "illness";
		else if(mention.equals("mobilised"))
			return "mobilize";
		else if(mention.equals("impactee"))
			return "impact";
		else if(mention.equals("genitor"))
			return "father";
		else
		    return mention;
	}

	
}
