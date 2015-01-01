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
import java.util.TreeSet;
import java.util.Map.Entry;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabTypeConverter;

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLChar;
import com.jmatio.types.MLCell;

import edu.pengli.nlp.conference.acl2015.types.Category;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.pipe.Noop;
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
	
	
	private HashMap<Instance, FeatureVector> instanceVectorMap;
	
	public FeatureVectorGenerator() throws IOException{
		instanceVectorMap = new HashMap<Instance, FeatureVector>();
	}
	
	
	protected Instance pipe(Instance inst) {
		
		FeatureVector fv = instanceVectorMap.get(inst);
		return new Instance(fv, null, null, (Pattern)inst.getSource());
	}
			
/*	public void batchGenerateVectorsByGeneralPatterns(String outputSummaryDir,
			String corpusName, InstanceList patternList, 
			MatlabProxy proxy) throws IOException, MatlabInvocationException{
		
		Alphabet dictionary = new Alphabet();
		int maxPatternSize = 0;
		InstanceList instances = new InstanceList(null);
		HashSet<String> set = new HashSet<String>();
		HashMap<Instance, Instance> tmpMap = new HashMap<Instance, Instance>();
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			CoreMap sentence = p.getAnnotatedSentence();
			
			SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			StringBuilder negativePattern = new StringBuilder();
			for (int i = 0; i < labels.size() - 1; i++) {
				CoreLabel nextToken = labels.get(i+1);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(nextToken.index());
				if(p.getArg1().contains(word) || p.getRel().contains(word)
						|| p.getArg2().contains(word))
					continue;
				if(word != null){
					negativePattern.append(word.originalText()+" ");
					dictionary.lookupIndex(word.originalText());	
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
			for(IndexedWord iw : p.getArg1()){
				size++;
				dictionary.lookupIndex(iw.originalText());	
			}
			String arg1Type = p.getArg1().getHead().ner().toUpperCase();
			arg1Type = arg1Type.replaceAll(" ", "_");
			positivePattern.append(arg1Type+" ");
			dictionary.lookupIndex(arg1Type);
			
			for(IndexedWord iw : p.getRel()){
				size++;
				dictionary.lookupIndex(iw.originalText());
			}
			positivePattern.append(p.getRel().getHead().originalText()+" ");
			dictionary.lookupIndex(p.getRel().getHead().originalText());

			for(IndexedWord iw : p.getArg2()){
				size++;
				dictionary.lookupIndex(iw.originalText());
			}		

			String arg2Type = p.getArg2().getHead().ner().toUpperCase();
			arg2Type = arg2Type.replaceAll(" ", "_");
			positivePattern.append(arg2Type+" ");
			dictionary.lookupIndex(arg2Type);
			
			Instance positiveInst = new Instance(positivePattern.toString().trim(), 
					"1", "positivePattern", sentence);
			tmpMap.put(positiveInst, instance);
			instances.add(positiveInst);
			if(size >= maxPatternSize)
				maxPatternSize = size;
		}
		
		ArrayList<String[]> training = new ArrayList<String[]>();
		ArrayList<String[]> validating = new ArrayList<String[]>();
		ArrayList<String[]> testing  = new ArrayList<String[]>();
		ArrayList<String[]> allPositives  = new ArrayList<String[]>();
		ArrayList<Instance> allPositives_L  = new ArrayList<Instance>();
		for(Instance inst: instances){
			String label = (String)inst.getTarget();
			if(label.equals("1")){
				String[] map = new String[2];
				map[0] = (String)inst.getData();
				map[1] = "1";
				allPositives.add(map);
				allPositives_L.add(inst);
			}	
		}
		
		if(allPositives.size() != patternList.size()){
			System.out.println("match error");
			System.exit(0);
		}
		
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
		ArrayList list2 = new ArrayList();
		list.add(cell);
		list2.add(cell);
		list.add(new MLDouble("sent_length", sentLength_arr, 1));
		list2.add(new MLDouble("sent_length", sentLength_arr, 1));
		list.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list2.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list.addAll(generateMatlabInput(testing, "test", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(testing, "test", maxPatternSize, dictionary));
		list.addAll(generateMatlabInput(training, "train", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(training, "train", maxPatternSize, dictionary));	
		list.addAll(generateMatlabInput(validating, "valid", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(allPositives, "valid", maxPatternSize, dictionary));
		

		int dim = 50;
		double[] vec = new double[dim*dictionary.size()];
		int c = 0;
		for(int i=0; i<dictionary.size(); i++){
			String word = (String)dictionary.lookupObject(i);
			float[] wordVector = wordMap.get(word);
			if(wordVector == null){
				for (int a = 0; a < dim; a++) {
					vec[c++] = rand.nextDouble();
				}
			}else{
				for (int a = 0; a < dim; a++) {
					vec[c++] = wordVector[a];
				}
			}
		
		}
		list.add(new MLDouble("vocab_emb", vec, dim));
		list2.add(new MLDouble("vocab_emb", vec, dim));
		String matInputFile = outputSummaryDir + "/" + corpusName + "_In.mat";
		String matInputFile2 = outputSummaryDir + "/" + corpusName + "_In2.mat";
		String modelOutputFile = outputSummaryDir + "/" + corpusName + "_Model.mat";
		String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out.mat";
		new MatFileWriter(matInputFile, list);
		new MatFileWriter(matInputFile2, list2);
		
		ArrayList<FeatureVector> fvs = getSentenceVectors(matInputFile, matInputFile2, 
				modelOutputFile, matOutputFile, proxy);
		
		if(fvs.size() != allPositives.size()){
			System.out.println("fv size is not equal all Positives size");
			System.exit(0);
		}

		instanceVectorMap = new HashMap<Instance, FeatureVector>();
		for(int i=0; i<allPositives.size(); i++){
			instanceVectorMap.put(tmpMap.get(allPositives_L.get(i)), fvs.get(i));
		}
	}*/
	
	public void trainingDCNN(String outputSummaryDir, InstanceList patternList, 
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
		
		HashMap<String, float[]> wordMap = initializeWordVectorMap();
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
	
	public static ArrayList<FeatureVector> 
	batchGetSeedVectorsForClustering(String outputSummaryDir,  
			Map<String, String[]> aspects
			, MatlabProxy proxy) throws IOException, ClassNotFoundException, MatlabInvocationException{
		
		ObjectInputStream in;

		in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + "ALL" + ".dict.ser"));
		Alphabet dictionary = (Alphabet)in.readObject();

		int maxPatternSize = in.readInt();
		in.close();


		Set<String> keys = aspects.keySet();
		ArrayList<String> candidates  =new ArrayList<String>();
		for (String key : keys) {
			String[] keywords = aspects.get(key);
			Set<String> removedDup = new HashSet<String>();
			for(String s: keywords){	
				removedDup.add(s);
			}
			StringBuilder positivePattern = new StringBuilder();
			int c = 0;
			for(String s: removedDup){
				if(c++ > 6)break;
				s = s.replaceAll(" ", "_");
				positivePattern.append(s+" ");
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
		
		String matInputFile = outputSummaryDir + "/" + "ALL" + "_In.mat";
		MatFileReader red = new MatFileReader(matInputFile);

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
		
		String matInputFile_AllPosi = outputSummaryDir + "/" + "_In_Seed.mat";
	
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model.mat";
		String matOutputFile = outputSummaryDir + "/" + "_Out_Seed.mat";
		
		
		new MatFileWriter(matInputFile_AllPosi, list);
			
		ArrayList<FeatureVector> fvs = FeatureVectorGenerator.
					getInstancesVectors(modelOutputFile, matInputFile_AllPosi,
					matOutputFile, proxy);

	
        return fvs;
		
	}
	
	public void batchGetVectorsForClustering(String outputSummaryDir, String corpusName, InstanceList patternList
			, MatlabProxy proxy) throws 
	FileNotFoundException, IOException, ClassNotFoundException, MatlabInvocationException{
		
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + "ALL" + ".dict.ser"));
		Alphabet dictionary = (Alphabet)in.readObject();

		int maxPatternSize = in.readInt();
		in.close();
		
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
		
		String matInputFile = outputSummaryDir + "/" + "ALL" + "_In.mat";
		MatFileReader red = new MatFileReader(matInputFile);
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
		new MatFileWriter(matInputFile_AllPosi, list);
		
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model.mat";
		String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out_AllPosi.mat";
		
		
		ArrayList<FeatureVector> fvs = FeatureVectorGenerator.
				getInstancesVectors(modelOutputFile, matInputFile_AllPosi,
				matOutputFile, proxy);
	
		for(int i=0; i<patternList.size(); i++){
			instanceVectorMap.put(patternList.get(i), fvs.get(i));
		}
		
	}
	
	public void batchGenerateVectorsBySpecificPatterns(String outputSummaryDir,
			String corpusName, InstanceList patternList, 
			MatlabProxy proxy) throws IOException, MatlabInvocationException{
		
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" +corpusName + ".dict.ser"));
		
		Alphabet dictionary = new Alphabet();
		int maxPatternSize = 0;
		InstanceList instances = new InstanceList(null);
		HashSet<String> set = new HashSet<String>();
		HashMap<Instance, Instance> tmpMap = new HashMap<Instance, Instance>();
		
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
			tmpMap.put(positiveInst, instance);
			instances.add(positiveInst);
			if(size >= maxPatternSize)
				maxPatternSize = size;
		}
		
		ArrayList<String[]> training = new ArrayList<String[]>();
		ArrayList<String[]> validating = new ArrayList<String[]>();
		ArrayList<String[]> testing  = new ArrayList<String[]>();
		ArrayList<String[]> allPositives  = new ArrayList<String[]>();
		ArrayList<Instance> allPositives_L  = new ArrayList<Instance>();
		for(Instance inst: instances){
			String label = (String)inst.getTarget();
			if(label.equals("1")){
				String[] map = new String[2];
				map[0] = (String)inst.getData();
				map[1] = "1";
				allPositives.add(map);
				allPositives_L.add(inst);
			}	
		}
		
		if(allPositives.size() != patternList.size()){
			System.out.println("match error");
			System.exit(0);
		}
		
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
		ArrayList list2 = new ArrayList();
		list.add(cell);
		list2.add(cell);
		list.add(new MLDouble("sent_length", sentLength_arr, 1));
		list2.add(new MLDouble("sent_length", sentLength_arr, 1));
		list.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list2.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list.addAll(generateMatlabInput(testing, "test", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(testing, "test", maxPatternSize, dictionary));
		list.addAll(generateMatlabInput(training, "train", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(training, "train", maxPatternSize, dictionary));	
		list.addAll(generateMatlabInput(validating, "valid", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(allPositives, "valid", maxPatternSize, dictionary));
		

		int dim = 50;
		double[] vec = new double[dim*dictionary.size()];
		int c = 0;
		HashMap<String, float[]> wordMap = initializeWordVectorMap();
		for(int i=0; i<dictionary.size(); i++){
			String word = (String)dictionary.lookupObject(i);
			float[] wordVector = wordMap.get(word);
			if(wordVector == null){
				for (int a = 0; a < dim; a++) {
					vec[c++] = rand.nextDouble();
				}
			}else{
				for (int a = 0; a < dim; a++) {
					vec[c++] = wordVector[a];
				}
			}
		
		}
		list.add(new MLDouble("vocab_emb", vec, dim));
		list2.add(new MLDouble("vocab_emb", vec, dim));
		String matInputFile = outputSummaryDir + "/" + corpusName + "_In.mat";
		String matInputFile2 = outputSummaryDir + "/" + corpusName + "_In2.mat";
		String modelOutputFile = outputSummaryDir + "/" + corpusName + "_Model.mat";
		String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out.mat";
		new MatFileWriter(matInputFile, list);
		new MatFileWriter(matInputFile2, list2);
		
		proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
				+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
		proxy.eval("Train('"+matInputFile+"', '"+modelOutputFile+"')");
		
		ArrayList<FeatureVector> fvs = getInstancesVectors( 
				modelOutputFile, matInputFile2, matOutputFile, proxy);
		
		if(fvs.size() != allPositives.size()){
			System.out.println("fv size is not equal all Positives size");
			System.exit(0);
		}

		for(int i=0; i<allPositives.size(); i++){
			instanceVectorMap.put(tmpMap.get(allPositives_L.get(i)), fvs.get(i));
		}
		
		out.writeObject(dictionary);
		out.writeInt(maxPatternSize);
		out.close();
	}
	
/*	public void batchGenerateVectorsByTuples(String outputSummaryDir,
			String corpusName, InstanceList patternList, 
			MatlabProxy proxy) throws IOException, MatlabInvocationException{
			
		Alphabet dictionary = new Alphabet();
		int maxPatternSize = 0;
		InstanceList instances = new InstanceList(null);
		HashSet<String> set = new HashSet<String>();
		HashMap<Instance, Instance> tmpMap = new HashMap<Instance, Instance>();
		for(Instance instance : patternList){
			Pattern p = (Pattern)instance.getSource();
			CoreMap sentence = p.getAnnotatedSentence();
			
			SemanticGraph graph = sentence.get(BasicDependenciesAnnotation.class);
			List<CoreLabel> labels = sentence.get(TokensAnnotation.class);
			StringBuilder negativeTuple = new StringBuilder();
			for (int i = 0; i < labels.size() - 1; i++) {
				CoreLabel nextToken = labels.get(i+1);
				//token may contain punc, however graph may not contain punc, so word may be null
				IndexedWord word = graph.getNodeByIndexSafe(nextToken.index());
				if(p.getArg1().contains(word) || p.getRel().contains(word)
						|| p.getArg2().contains(word))
					continue;
				if(word != null){
					negativeTuple.append(word.originalText()+" ");
					dictionary.lookupIndex(word.originalText());	
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
			
			StringBuilder positiveTuple = new StringBuilder();
		
			int size = 0;
			for(IndexedWord iw : p.getArg1()){
				size++;
				dictionary.lookupIndex(iw.originalText());	
				positiveTuple.append(iw.originalText()+" ");
			}
			
			for(IndexedWord iw : p.getRel()){
				size++;
				dictionary.lookupIndex(iw.originalText());
				positiveTuple.append(iw.originalText()+" ");
			}

			for(IndexedWord iw : p.getArg2()){
				size++;
				dictionary.lookupIndex(iw.originalText());
				positiveTuple.append(iw.originalText()+" ");
			}		
			
			Instance positiveInst = new Instance(positiveTuple.toString().trim(), 
					"1", "positiveTuple", sentence);
			tmpMap.put(positiveInst, instance);
			instances.add(positiveInst);
			if(size >= maxPatternSize)
				maxPatternSize = size;
		}
		
		ArrayList<String[]> training = new ArrayList<String[]>();
		ArrayList<String[]> validating = new ArrayList<String[]>();
		ArrayList<String[]> testing  = new ArrayList<String[]>();
		ArrayList<String[]> allPositives  = new ArrayList<String[]>();
		ArrayList<Instance> allPositives_L  = new ArrayList<Instance>();
		for(Instance inst: instances){
			String label = (String)inst.getTarget();
			if(label.equals("1")){
				String[] map = new String[2];
				map[0] = (String)inst.getData();
				map[1] = "1";
				allPositives.add(map);
				allPositives_L.add(inst);
			}	
		}
		
		if(allPositives.size() != patternList.size()){
			System.out.println("match error");
			System.exit(0);
		}
		
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
		ArrayList list2 = new ArrayList();
		list.add(cell);
		list2.add(cell);
		list.add(new MLDouble("sent_length", sentLength_arr, 1));
		list2.add(new MLDouble("sent_length", sentLength_arr, 1));
		list.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list2.add(new MLDouble("size_vocab", vocSize_arr, 1));
		list.addAll(generateMatlabInput(testing, "test", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(testing, "test", maxPatternSize, dictionary));
		list.addAll(generateMatlabInput(training, "train", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(training, "train", maxPatternSize, dictionary));	
		list.addAll(generateMatlabInput(validating, "valid", maxPatternSize, dictionary));
		list2.addAll(generateMatlabInput(allPositives, "valid", maxPatternSize, dictionary));
		

		int dim = 50;
		double[] vec = new double[dim*dictionary.size()];
		int c = 0;
		for(int i=0; i<dictionary.size(); i++){
			String word = (String)dictionary.lookupObject(i);
			float[] wordVector = wordMap.get(word);
			if(wordVector == null){
				for (int a = 0; a < dim; a++) {
					vec[c++] = rand.nextDouble();
				}
			}else{
				for (int a = 0; a < dim; a++) {
					vec[c++] = wordVector[a];
				}
			}
		
		}
		list.add(new MLDouble("vocab_emb", vec, dim));
		list2.add(new MLDouble("vocab_emb", vec, dim));
		String matInputFile = outputSummaryDir + "/" + corpusName + "_In.mat";
		String matInputFile2 = outputSummaryDir + "/" + corpusName + "_In2.mat";
		String modelOutputFile = outputSummaryDir + "/" + corpusName + "_Model.mat";
		String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out.mat";
		new MatFileWriter(matInputFile, list);
		new MatFileWriter(matInputFile2, list2);
		
		ArrayList<FeatureVector> fvs = getSentenceVectors(matInputFile, matInputFile2, 
				modelOutputFile, matOutputFile, proxy);
		
		if(fvs.size() != allPositives.size()){
			System.out.println("fv size is not equal all Positives size");
			System.exit(0);
		}

		instanceVectorMap = new HashMap<Instance, FeatureVector>();
		for(int i=0; i<allPositives.size(); i++){
			instanceVectorMap.put(tmpMap.get(allPositives_L.get(i)), fvs.get(i));
		}
		

	}*/
	
	public static ArrayList<MLDouble> generateMatlabInput
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
		
	public static ArrayList<FeatureVector> getInstancesVectors(String modelOutputFile, String matInputFile, 
			String matOutputFile, MatlabProxy proxy) 
					throws MatlabInvocationException, FileNotFoundException, IOException{
		
		proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
				+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
		proxy.eval("MyScript('"+modelOutputFile+"',"+"'"+matInputFile+"',"+"'"+matOutputFile+"'"+")");

		
		MatFileReader red = new MatFileReader(matOutputFile);
		MLDouble data = (MLDouble)red.getMLArray("M_3");
		double[][] arr = data.getArray();
		int m = data.getM();
		int n = data.getN();
		
		ArrayList<FeatureVector> ret = new ArrayList<FeatureVector>();
		for(int i=0; i<m; i++){
			double[] vec = new double[n];
			int[] idx = new int[n];
			int c = 0;
			for(int j=0; j<n; j++){
				vec[c++] = arr[i][j];
				idx[j] = j;
			}
			FeatureVector fv = new FeatureVector(idx, vec);
			ret.add(fv);
		}
		return ret;
	}
	
	public static FeatureVector getKeywordsVector(String outputSummaryDir, 
			String[] keywords, int sizeofWordVector, MatlabProxy proxy){
		
		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model.mat";
		FeatureVector fv = null;
		
		try {
			
			ObjectInputStream in= new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + "ALL" + ".dict.ser"));
			
			Alphabet dictionary = (Alphabet)in.readObject();
		
			double[][] arr = new double[sizeofWordVector][dictionary.size()];
	
			proxy.eval("addpath('/home/peng/Develop/Workspace/Mavericks/platform/src/main/java/edu"
					+ "/pengli/nlp/platform/algorithms/neuralnetwork/DCNN')");
			proxy.eval("load('"+modelOutputFile+"')");
			proxy.eval("[CR_E, CR_1, CR_1_b, CR_2, CR_2_b, CR_3, CR_3_b, CR_Z, ~, ~] "
					+ "= stack2param(X, decodeInfo);");
			
			arr = processor.getNumericArray("CR_E")
					.getRealArray2D();
	
			int rowNumber = sizeofWordVector;
			int columnNumber = dictionary.size();
			double[] vec = new double[rowNumber];
			for(String word : keywords){
				word = word.replaceAll(" ", "_");
				int wordIdx = dictionary.lookupIndex(word, false);
				for(int j=0; j< columnNumber; j++){
					if(j == wordIdx){
						for(int i=0; i<rowNumber; i++)
							vec[i] += arr[i][j];
					}	
				}
			}
			
/*			for (int a = 0; a < rowNumber; a++) {
				vec[a] /= rowNumber;
			}*/
			
			int[] idx = new int[rowNumber];
			for(int a = 0; a < rowNumber; a++){
				idx[a] = a;
			}
			
			fv = new FeatureVector(idx, vec);
			
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return fv;
	}


	private HashMap<String, float[]> initializeWordVectorMap() throws IOException{
		System.out.println("Begin to load word vectors");
		
		int max_w = 50; // max length of vocabulary entries
        String modelPath = "/home/peng/Develop/Workspace/Mavericks/models"
        		+ "/word2vec/GoogleNews-vectors-negative300.bin";	
		HashMap<String, float[]> wordMap = new HashMap<String, float[]>();
		BufferedInputStream bis = new BufferedInputStream(new FileInputStream(
				modelPath));
		DataInputStream dis = new DataInputStream(bis);
		int words, size;
		double len;
		String firstLine = dis.readLine();
		words = Integer.parseInt(firstLine.split(" ")[0]);
		size = Integer.parseInt(firstLine.split(" ")[1]);
		int dimension = size;
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
		
		return wordMap;
		
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
	
}
