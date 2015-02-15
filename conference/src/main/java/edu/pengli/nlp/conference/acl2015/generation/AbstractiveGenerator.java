package edu.pengli.nlp.conference.acl2015.generation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;

import lpsolve.LpSolve;
import lpsolve.LpSolveException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.conference.acl2015.pipe.FeatureVectorGenerator;
import edu.pengli.nlp.conference.acl2015.types.Category;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.algorithms.classify.Clustering;
import edu.pengli.nlp.platform.algorithms.classify.HarmonicSemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.classify.KMeans_Java;
import edu.pengli.nlp.platform.algorithms.classify.KMeans_Matlab;
import edu.pengli.nlp.platform.algorithms.classify.LabelPropagationSemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.classify.SemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.miscellaneous.Merger;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;
import edu.pengli.nlp.platform.types.NormalizedDotProductMetric;
import edu.pengli.nlp.platform.types.SparseVector;
import edu.pengli.nlp.platform.util.CallableTask;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RankMap;
import edu.pengli.nlp.platform.util.TimeWait;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;

public class AbstractiveGenerator {
	
	public void NLGMethod(String inputCorpusDir, String outputSummaryDir,
			String corpusName, String categoryId, MatlabProxy proxy,  int sigma){
		
		/*		HashSet<Pattern> patternSet = null;
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + corpusName + ".patterns.ser"));
			patternSet = (HashSet<Pattern>) in.readObject();
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

		InstanceList patternList = new InstanceList(new Noop());
		for (Pattern p : patternSet) {
			Instance inst = new Instance(p, null, null, p);
			patternList.add(inst);
		}

		PipeLine pipeLine = new PipeLine();
		FeatureVectorGenerator fvg = new FeatureVectorGenerator();
		
		generateSmallWordVector (outputSummaryDir, corpusName,
		patternList, categoryId);
		

		fvg.setFvsViaPreTrainedWord2VecModel(outputSummaryDir,
		 corpusName, patternList);
		
		InstanceList patternInstances = new InstanceList(pipeLine);
		pipeLine.addPipe(fvg);
		patternInstances.addThruPipe(patternList.iterator());

		try {
			
			ObjectOutputStream out = new ObjectOutputStream(
					new FileOutputStream(outputSummaryDir + "/" + corpusName
							+ ".featuredInsts"));
			patternInstances.writeObject(out);
			out.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		

		InstanceList patternInstances = new InstanceList(new Noop());
		try {
			ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(outputSummaryDir
					+ "/" + corpusName + ".featuredInsts"));
			patternInstances.readObject(inInst);
			inInst.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);

		int length = 0;
		boolean flag = false;
		
//		System.out.println("Begin to train RNN to score patterns");
//		trainRNN(outputSummaryDir, corpusName);

		// 1 pattern Clustering
		InstanceList seeds = generateFVsForSeeds(outputSummaryDir, corpusName,
		 categoryId, proxy);
		
/*		SemiSupervisedClustering semiClustering = new 
		HarmonicSemiSupervisedClustering(
		new Noop(), seeds, new NormalizedDotProductMetric(), proxy, sigma);*/

/*		SemiSupervisedClustering semiClustering = new 
		LocalglobalConsistencySemiSupervisedClustering(
		new Noop(), seeds, new NormalizedDotProductMetric(), proxy, sigma);*/
		
		SemiSupervisedClustering semiClustering = new 
				LabelPropagationSemiSupervisedClustering(
				new Noop(), seeds, new NormalizedDotProductMetric(), proxy, sigma);
		
		Clustering predicted = semiClustering.cluster(patternInstances);
		
		InstanceList[] groups = predicted.getClusters();
		HashMap<InstanceList, Integer> tmp = new HashMap<InstanceList, Integer>();
		for (InstanceList il : groups)
			tmp.put(il, il.size());
		HashMap rankedMap = RankMap.sortHashMapByValues(tmp, false);
		Set keys = rankedMap.keySet();
		Iterator iter = keys.iterator();
		Metric metric = new NormalizedDotProductMetric();
		NLGGenerator nlg = new NLGGenerator();
		while (iter.hasNext()) {
			InstanceList il = (InstanceList) iter.next();
			if(il.size() == 0)
				continue;
			SparseVector clusterMean = KMeans_Java.mean(il);
			double mindist = Double.MAX_VALUE;
			int idx = -1;
			for(int i=0; i<il.size(); i++){
				double dist = metric.distance(clusterMean, 
						(FeatureVector)il.get(i).getData());
				if(dist < mindist){
					idx = i;
					mindist = dist;
				}
			}
			
			Pattern p = (Pattern)il.get(idx).getSource();
			SemanticGraph graph = p.getTuple().
					getAnnotatedSentence().get(BasicDependenciesAnnotation.class);
			String line = nlg.realization(p, graph);
			if (line == null)
				continue;

			String[] toks = line.split(" ");
			length += toks.length;
			if (length <= 100)
				out.println(line);
			else {
				length -= toks.length;
				boolean stop = false;
				for (int i = 0; i < toks.length; i++) {
					out.print(toks[i] + " ");
					length++;
					if (length > 100) {
						stop = true;
						break;
					}
				}
				if (stop == true) {
					flag = true;
					break;
				}
			}
			
			if (flag == true)
				break;
		}
		
		out.close();
		
	}

	public void sigirMethod(String inputCorpusDir, String outputSummaryDir,
			String corpusName, String categoryId, MatlabProxy proxy, int sigma) {

/*		HashSet<Pattern> patternSet = null;
		try {
			ObjectInputStream in = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + corpusName + ".patterns.ser"));
			patternSet = (HashSet<Pattern>) in.readObject();
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

		InstanceList patternList = new InstanceList(new Noop());
		for (Pattern p : patternSet) {
			Instance inst = new Instance(p, null, null, p);
			patternList.add(inst);
		}

		PipeLine pipeLine = new PipeLine();
		FeatureVectorGenerator fvg = new FeatureVectorGenerator();
		
		generateSmallWordVector (outputSummaryDir, corpusName,
		patternList, categoryId);
		

		fvg.setFvsViaPreTrainedWord2VecModel(outputSummaryDir,
		 corpusName, patternList);
		
		InstanceList patternInstances = new InstanceList(pipeLine);
		pipeLine.addPipe(fvg);
		patternInstances.addThruPipe(patternList.iterator());

		try {
			
			ObjectOutputStream out = new ObjectOutputStream(
					new FileOutputStream(outputSummaryDir + "/" + corpusName
							+ ".featuredInsts"));
			patternInstances.writeObject(out);
			out.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		

		InstanceList patternInstances = new InstanceList(new Noop());
		try {
			ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(outputSummaryDir
					+ "/" + corpusName + ".featuredInsts"));
			patternInstances.readObject(inInst);
			inInst.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);

		int length = 0;
		boolean flag = false;
		
//		System.out.println("Begin to train RNN to score patterns");
//		trainRNN(outputSummaryDir, corpusName);

		// 1 pattern Clustering
		InstanceList seeds = generateFVsForSeeds(outputSummaryDir, corpusName,
		 categoryId, proxy);
		
/*		SemiSupervisedClustering semiClustering = new 
		HarmonicSemiSupervisedClustering(
		new Noop(), seeds, new NormalizedDotProductMetric(), proxy, sigma);*/

/*		SemiSupervisedClustering semiClustering = new 
		LocalglobalConsistencySemiSupervisedClustering(
		new Noop(), seeds, new NormalizedDotProductMetric(), proxy, sigma);*/
		
		SemiSupervisedClustering semiClustering = new 
				LabelPropagationSemiSupervisedClustering(
				new Noop(), seeds, new NormalizedDotProductMetric(), proxy, sigma);
		
		Clustering predicted = semiClustering.cluster(patternInstances);
		
		HashMap<Integer, String> labelAspectMap = new HashMap<Integer, String>();
		for(int i=0; i<seeds.size(); i++){
			labelAspectMap.put(i, (String)seeds.get(i).getSource());
		}
		
		InstanceList[] groups_seed = predicted.getClusters();
		Category[] cats = Category.values();
		for (Category cat : cats) {
			if (cat.getId() == Integer.parseInt(categoryId)) {
				
				Map<String, String[]> aspects = cat.getAspects(cat.getId());
				Set<String> aspectsSet = aspects.keySet();
				for (String aspect : aspectsSet) {
					ArrayList<InstanceList> localClusters = 
							new ArrayList<InstanceList>();
					
					for(int i=0; i<groups_seed.length; i++){
						if(labelAspectMap.get(i).equals(aspect)){
							localClusters.add(groups_seed[i]);
						}
					}
					
					HashMap<InstanceList, Integer> tmp = new HashMap<InstanceList, Integer>();
					for (InstanceList il : localClusters)
						tmp.put(il, il.size());
					
					HashMap rankedMap = RankMap.sortHashMapByValues(tmp, false);
					Set keys = rankedMap.keySet();
					Iterator iter = keys.iterator();
					while (iter.hasNext()) {
						InstanceList il = (InstanceList) iter.next();
						String sent = realization(outputSummaryDir, corpusName,
								il, proxy);
						if (sent == null)
							continue;

						String[] toks = sent.split(" ");
						length += toks.length;
						if (length <= 100){
							out.println(sent);
						}
							
						else {
							length -= toks.length;
							boolean stop = false;
							for (int i = 0; i < toks.length; i++) {
								out.print(toks[i] + " ");
								length++;
								if (length > 100) {
									stop = true;
									break;
								}
							}
							if (stop == true) {
								flag = true;
								break;
							}
						}	
						

						if (flag == true)
							break;
					}
					
				}	
				
			}//end if find aspects
			
		}
		
		out.close();
		
	}
	
	public void ijcaiMethod(String inputCorpusDir, String outputSummaryDir,
			String corpusName, int topN, MatlabProxy proxy) throws LpSolveException{
		
		
		//1. find representative patterns as seeds
//		System.out.println("find representative patterns as seeds");
		int nIter = 7;
		InstanceList instances = new InstanceList(new Noop());
		Metric metric = new NormalizedDotProductMetric();
		
		try {
			ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + corpusName + ".featuredInsts"));
			instances.readObject(inInst);
			inInst.close();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	
		
		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
		
/*		FeatureVector fv_0 = (FeatureVector) instances.get(0).getData();
		double[][] dataMatrix = new double[fv_0.getIndices().length][instances
		                        				.size()];
		                        		
		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
		    double[] vals = fv_i.getValues();
		    for(int j=0; j< vals.length; j++)
		    dataMatrix[j][i] = vals[j];

		}
		                        		
		double[][] X0 = new double[instances.size()][instances.size()];
		Random rand = new Random();     		
		for (int i = 0; i < instances.size(); i++) {
			for(int j=0; j<instances.size(); j++)
				X0[i][j]= rand.nextDouble();
		}
		                        			
		try {
				
			processor.setNumericArray("A", new MatlabNumericArray(
					dataMatrix, null));
			
			processor.setNumericArray("Y", new MatlabNumericArray(
					dataMatrix, null));
			
			processor.setNumericArray("X0", new MatlabNumericArray(
					X0, null));

			proxy.eval("n = size(A,2)");
			proxy.eval("p = size(A,2)");
			proxy.eval("Xi = sqrt(sum(X0.*X0,2)+eps)");
			proxy.eval("d = 0.5./(Xi)");
			proxy.eval("AX = A*X0-Y");
			proxy.eval("Xi1 = sqrt(sum(AX.*AX,1)+eps)");
			proxy.eval("d1 = 0.5./Xi1");
			proxy.eval("AA = A'*A");
			proxy.eval("AY = A'*Y");
			
			proxy.eval("X=zeros(n,n)");
			for(int i=1; i<=nIter; i++){
				proxy.eval("D = spdiags(d,0,n,n)");
				for(int j=1; j<=instances.size(); j++){
					proxy.eval("X(:,"+j+") = mldivide((d1("+j+")*AA+"+
				parameter+"*D),(d1("+j+")*AY(:,"+j+")))");	
				}
				proxy.eval("Xi = sqrt(sum(X.*X,2)+eps)");
				proxy.eval("d = 0.5./Xi");
				proxy.eval("AX = A*X-Y");
				proxy.eval("Xi1 = sqrt(sum(AX.*AX,1)+eps)");
				proxy.eval("d1 = 0.5./Xi1");
//				proxy.eval("obj("+i+") = (sum(Xi1) + r*sum(Xi))");
			}
			
			proxy.eval("a = sum(abs(X),2)");
			proxy.eval("[rank_value, rank_idx] = sort(a,'descend')");
		
		} catch (MatlabInvocationException e) {
			e.printStackTrace();
		}
		
		double[][] rank_idx = processor.getNumericArray("rank_idx").getRealArray2D();  
		InstanceList patternCluster = new InstanceList(pipeLine);
		for(int i=0; i<topN; i++){
			patternCluster.add(instances.get((int)(rank_idx[i][0]-1)));
		}
		
		ObjectOutputStream outP = new ObjectOutputStream(new FileOutputStream(
				 outputSummaryDir + "/" +corpusName + ".patternCluster."+parameter+"."+topN)); 
		patternCluster.writeObject(outP);
		outP.close();*/
		
		InstanceList patternCluster = new InstanceList(new Noop());
		int parameter = 10;
		try {
			ObjectInputStream inPatternCluster = new ObjectInputStream(new FileInputStream(
					outputSummaryDir + "/" + corpusName + ".patternCluster."+parameter));
			patternCluster.readObject(inPatternCluster);
			inPatternCluster.close();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
			
		//2, clustering tuples with LGC semi supervised learning
//		System.out.println("clustering tuples with LGC semi supervised learning");
		SemiSupervisedClustering semiClustering = new 
		HarmonicSemiSupervisedClustering(
		new Noop(), patternCluster, metric, proxy, 20);
		Clustering clusters = semiClustering.cluster(instances);
				
/*		FeatureVector vec = (FeatureVector)instances.get(0).getData();
		int dimension = vec.getValues().length;
		
		double[][] dataMatrix = new double[instances.size()][dimension];
		for (int i = 0; i < instances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) instances.get(i).getData();
			for(int j=0; j<dimension; j++)
				dataMatrix[i][j] = fv_i.getValues()[j];
		}
		
		int clusterLabels[] = new int[instances.size()];
		try {
			
			processor.setNumericArray("arr", new MatlabNumericArray(
					dataMatrix, null));
			proxy.eval("labels = kmeans(arr,"+topN+ ",'Replicates',20)");
			double[][] labels = processor.getNumericArray("labels").getRealArray2D();
			for (int i = 0; i < instances.size(); i++)
				clusterLabels[i] = (int) labels[i][0]-1;
			
		} catch (MatlabInvocationException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		Clustering clusters = new Clustering(instances, topN, clusterLabels);*/
		
		
/*		Spectral spectral = new Spectral(new Noop(), topN, metric, proxy);
		Clustering clusters = spectral.cluster(instances);*/
	
		
		//3, begin to fuse tuples and rank new tuples in each clusters
//		System.out.println("fuse and rank tuples");

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
	
		InstanceList[] groups = clusters.getClusters();
		
		InstanceList rankedInstanceList = new InstanceList(new Noop());
		ArrayList<Integer> labelsList = new ArrayList<Integer>();
		int clusterIdx = 0;
		int labelIdx = 0;
		for(int i=0; i<groups.length; i++){
			Instance patternInst = patternCluster.get(i);
			String bestPattern= (String) patternInst.getSource().toString();
			InstanceList group = groups[i];
			ArrayList<String> tupleCandidates = 
					generateSentenceByFusion(group, "tuple");
			if(tupleCandidates == null)
				return;
			
			HashMap<Instance, Double> nbestMap_tuple = getNbestMap(outputSummaryDir,
					corpusName, tupleCandidates);
			LinkedHashMap rankedmap_t = RankMap.sortHashMapByValues(nbestMap_tuple, true);
			HashMap<Instance, Double> nbestMap_tuple_N = new HashMap<Instance, Double>();
			Set<Instance> ks = rankedmap_t.keySet();
			Iterator iks = ks.iterator();
			double rank = 1.0;
			while(iks.hasNext()){
				Instance ins = (Instance)iks.next();
				nbestMap_tuple_N.put(ins, 1/(rank++));
			}
			
			HashMap<Instance, Double> tupleScoreMap = new HashMap<Instance, Double>();

			for(String tuple : tupleCandidates){
				String[] tokTup = tuple.split(" ");
				double[] vec_T = new double[300];
				int[] idx_T = new int[300];
				for(int a = 0; a <300; a++){
					idx_T[a] = a;
				}
				
				for(String tokT : tokTup){
					tokT = cleaning(tokT.toLowerCase());
					float[] wordVector_T = wordMap.get(tokT);
					
					if (wordVector_T == null)
							continue;
					for (int a = 0; a < 300; a++) {
						vec_T[a] += wordVector_T[a];
					}	
					
					float len = 0;
					for (int a = 0; a < 300; a++) {
						len += vec_T[a] * vec_T[a];
					}
					len = (float) Math.sqrt(len);
					for (int a = 0; a < 300; a++) {
						vec_T[a] /= len;
					}
					
				}
					

				
				double coverageScore = 0.0;
				
				String[] tokPat = bestPattern.split(" ");
				for(String tokT : tokTup)
					for(String tokP : tokPat){
						tokP = cleaning(tokP.toLowerCase());
						float[] wordVector_P = wordMap.get(tokP);
						
						tokT = cleaning(tokT.toLowerCase());
						float[] wordVector_T = wordMap.get(tokT);
										
						if(wordVector_P != null && wordVector_T != null){
							int[] idx = new int[wordVector_P.length];
							for (int a = 0; a < wordVector_P.length; a++) {
								idx[a] = a;
							}
							double[] wordVector_T_D = new double[wordVector_T.length];
							double[] wordVector_P_D = new double[wordVector_P.length];
							for(int a = 0; a < wordVector_T.length; a++){
								wordVector_T_D[a] = wordVector_T[a];
								wordVector_P_D[a] = wordVector_P[a];
							}
								
							FeatureVector fv_t = new FeatureVector(idx, wordVector_T_D);
							FeatureVector fv_p = new FeatureVector(idx, wordVector_P_D);
		
							coverageScore += (1- metric.distance(fv_t, fv_p));
						}
						
					}
				
				double fluencyScore = 0.0;
				Set<Instance> set = nbestMap_tuple_N.keySet();
				Iterator iterKey = set.iterator();
				while(iterKey.hasNext()){
					Instance ins = (Instance)iterKey.next();
					if(tuple.equals((String)ins.getData())){
						fluencyScore = nbestMap_tuple_N.get(ins);
						break;
					}
				}
				
				fluencyScore = 1/(1+Math.exp(-fluencyScore));
				coverageScore = 1/(1+Math.exp(-coverageScore));
				double score = 0.7*coverageScore + 0.3*fluencyScore;
				
				FeatureVector fv = new FeatureVector(idx_T, vec_T);
				Instance tupleInst = new Instance(fv, null, null, tuple);		
				tupleScoreMap.put(tupleInst, score);
			}
			
			HashMap sortedScores = RankMap.sortHashMapByValues(tupleScoreMap, false);
			Set<String> ts = sortedScores.keySet();
			Iterator its = ts.iterator();
			while(its.hasNext() ){
				Instance tupleInst = (Instance)its.next();
				rankedInstanceList.add(tupleInst);
				labelsList.add(labelIdx++, clusterIdx);
			}
			
			clusterIdx++;
				
		}
	
		int[] labels = new int[labelsList.size()];
		for(int i=0; i<labels.length; i++)
			labels[i] = labelsList.get(i);
		Clustering rankedClusters = new Clustering(rankedInstanceList,
				topN, labels);
		
		//4, generate final summary with simple greedy method
/*		double[][] matrix = new double[patternCluster.size()][rankedInstanceList.size()];
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);
		for(int label = 0; label < patternCluster.size(); label++){
			int[] idxs = rankedClusters.getIndicesWithLabel(label);
			if(idxs.length == 0){
				for(int i=0; i<rankedInstanceList.size(); i++)
						matrix[label][i] = -1;		
			}else{
				for(int i=0; i<rankedInstanceList.size(); i++){
					if(i < idxs.length)
						matrix[label][i] = idxs[i];
					else
						matrix[label][i] = -1;
				}
			}
		}
		
		Matrix Mat = new Matrix(matrix);
		Matrix transMat = Mat.transpose();
		
		HashSet<String> dup = new HashSet<String>();
		int length = 0;
		boolean flag = false;
		for(int i=0; i< rankedInstanceList.size(); i++){
			for(int j = 0; j<patternCluster.size(); j++){
				double idx = transMat.get(i, j);
				if(idx != -1){
					Instance tuple = rankedInstanceList.get((int)idx);
					String line = (String)tuple.getSource();
					boolean included = true;
					if(dup.size() != 0 ){
						for(String s : dup){
							double sim = LongestCommonSubstring.getSim(s, line);
							if(sim > threshold)
								included = false;
						}
					}
					
					if(!dup.contains(line) && included){
						dup.add(line);
					}else
						continue;
					String[] toks = line.split(" ");
					length += toks.length;
					if (length <= 200)
						out.println(line);
					else {
						length -= toks.length;
						boolean stop = false;
						for (int k = 0; k < toks.length; k++) {
							out.print(toks[k] + " ");
							length++;
							if (length > 200) {
								stop = true;
								break;
							}
						}
						if (stop == true) {
							flag = true;
							break;
						}
					}
					
					if (flag == true)
						break;
				}
			}
		}
		
		out.close();*/
		
		//begin ILP
//		System.out.println("ILP");
		LpSolve solver = LpSolve.makeLp(0, rankedClusters.getInstances().size());
		solver.setOutputfile("");

		for (int i = 0; i < rankedClusters.getInstances().size(); i++) {
			solver.setColName(i + 1, "s" + i);
			solver.setBinary(i + 1, true);
		}

		//set Objective Function
		StringBuffer sb_o = new StringBuffer();
		InstanceList[] cs = rankedClusters.getClusters();
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				int posi = l + 1;
				sb_o.append(posi + " ");
			}
		}

		solver.strSetObjFn(sb_o.toString().trim());
		solver.setMinim();

		// length constraints
		StringBuffer sb_L_N = new StringBuffer();
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				Instance sent = cluster.get(l);
				String sentMention = (String) sent.getSource();
				sb_L_N.append(sentMention.split(" ").length + " ");

			}

		}

		solver.strAddConstraint(sb_L_N.toString().trim(), LpSolve.LE,
				150);

		// exclusivity constraints
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		int globalIdx = 0;
		for (int c = 0; c < cs.length; c++) {
			InstanceList cluster = cs[c];
			for (int l = 0; l < cluster.size(); l++) {
				map.put(c + "_" + l, globalIdx++);
			}
		}

		int start = 0;
		for (int g = 0; g < cs.length; g++) {
			StringBuffer sb_E_N = new StringBuffer();
			for (int c = 0; c < cs.length; c++) {
				InstanceList cluster = cs[c];
				for (int k = 0; k < cluster.size(); k++) {
					int idx = map.get(c + "_" + k);
					if (idx >= start && idx <= start + cluster.size() - 1) {
						sb_E_N.append(1 + " ");
					} else
						sb_E_N.append(0 + " ");

				}

			}
			start += cs[g].size();
			solver.strAddConstraint(sb_E_N.toString().trim(), LpSolve.EQ, 1);
		}

/*		// Redundancy Constraints
		for (int i = 0; i < cs.length; i++) {
			InstanceList clusteri = cs[i];
			for (int m = 0; m < clusteri.size(); m++) {
				for (int j = i + 1; j < cs.length; j++) {
					InstanceList clusterj = cs[j];
					for (int n = 0; n < clusterj.size(); n++) {
						solver.strAddConstraint(
								buildStrVector(m, n, i, j, 
										rankedClusters, rankedInstanceList.size(), metric),
								LpSolve.LE, 0.1);
					}
				}
			}
		}*/

		solver.solve();

		solver.setVerbose(LpSolve.IMPORTANT);
		double[] var = solver.getPtrVariables();
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);
		for (int i = 0; i < var.length; i++) {
			if (var[i] == 1.0) {
				Instance sent = rankedClusters.getInstances().get(i);
				out.println(sent.getSource());
			}
		}
		out.close();
	}
	
	private void trainRNN(String outputSummaryDir, String corpusName)
			throws IOException {
		PrintWriter out_valid = new PrintWriter(new FileOutputStream(new File(
				outputSummaryDir + "/" + corpusName + ".patterns.valid")));

		BufferedReader in_train = new BufferedReader(new FileReader(new File(
				outputSummaryDir + "/" + corpusName + ".patterns")));
		ArrayList<String> trainsents = new ArrayList<String>();
		String input = null;
		while ((input = in_train.readLine()) != null) {
			trainsents.add(input);
		}
		in_train.close();
		Random rand = new Random();
		int size = trainsents.size();
		int newSize = size;
		for (int i = 0; i < size * 0.2; i++) {
			int ran = rand.nextInt(newSize);
			out_valid.println(trainsents.get(ran));
			trainsents.remove(ran);
			newSize--;
		}
		out_valid.close();
		String[] cmd = {
				"/home/peng/Develop/Workspace/Mavericks/platform/src"
						+ "/main/java/edu/pengli/nlp/platform/algorithms/neuralnetwork/RNNLM/rnnlm",
				"-train", outputSummaryDir + "/" + corpusName + ".patterns",
				"-valid",
				outputSummaryDir + "/" + corpusName + ".patterns.valid",
				"-rnnlm", outputSummaryDir + "/" + corpusName + ".rnnlm.model",
				"-hidden", "40", "-rand-seed", "1", "-debug", "2", "-bptt",
				"3", "-class", "200" };

		Process proc = Runtime.getRuntime().exec(cmd);
		try {

			while (proc.waitFor() != 0) {
				TimeWait.waiting(100);
			}
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	private InstanceList generateFVsForSeeds(String outputSummaryDir, String corpusName,
			String categoryId, MatlabProxy proxy){
		
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

		Metric metric = new NormalizedDotProductMetric();
		KMeans_Matlab km = new KMeans_Matlab(new Noop(), 4, metric,proxy);
		
		InstanceList seeds = new InstanceList(new Noop());
		Category[] cats = Category.values();
		int dimension = 300;
		for (Category cat : cats) {
			if (cat.getId() == Integer.parseInt(categoryId)) {
				Map<String, String[]> aspects = cat.getAspects(cat.getId());
				Set<String> keys = aspects.keySet();
				for (String k : keys) {

					String[] words = aspects.get(k);
					InstanceList wordsList = new InstanceList(null);
					for (String word : words) {
						double[] vec = new double[dimension];
						int[] idx = new int[dimension];
						
						float[] wordVector = wordMap.get(word);
						if (wordVector == null)
							continue;
						for (int a = 0; a < dimension; a++) {
							vec[a] = wordVector[a];
							idx[a] = a;
						}
						
						FeatureVector fv = new FeatureVector(idx, vec);
						Instance wordInst = new Instance(fv, null, null, word);
						wordsList.add(wordInst);
					}
					
					InstanceList[] groups = km.cluster(wordsList).getClusters();
					for(InstanceList il : groups){
						double[] vec = new double[dimension];
						int[] idx = new int[dimension];
						
						for(Instance inst : il){
							String word = (String)inst.getSource();
							float[] wordVector = wordMap.get(word);
							if (wordVector == null)
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
						
						for(int a = 0; a <dimension; a++){
							idx[a] = a;
						}
						
						double[] vec3 = new double[dimension*3];
						int[] idx3 = new int[dimension*3];
						
						int c=0;
						for(int i=0; i<dimension*3; i++){
							vec3[i] = vec[c++];
							if(c==dimension)
								c = 0;
						}
						
						int d = 0;
						for(int i=0; i<dimension*3; i++){
							idx3[i] = d++;
						}
						
						FeatureVector fv = new FeatureVector(idx3, vec3);
						Instance seed = new Instance(fv, null, null, k);
						seeds.add(seed);
					}
				}
			}
		}
		
		return seeds;
	}
	
	private String realization(String outputSummaryDir,
			String corpusName, InstanceList patternCluster, MatlabProxy proxy){
		
		//1. find the best pattern
		ArrayList<String> tupleCandidates = generateSentenceByFusion(patternCluster, "tuple");
		if(tupleCandidates == null)
			return null;
		ArrayList<String> patternCandidates = generateSentenceByFusion(patternCluster, "pattern");	
		if(patternCandidates == null)
			return null;
		if (tupleCandidates.size() == 0 || patternCandidates.size() == 0) {
			System.out.println("tuple or pattern set is empty");
			return null;
		}
		
		HashMap<Instance, Double> nbestMap_pattern = getNbestMap(outputSummaryDir,
				corpusName, patternCandidates);
		LinkedHashMap rankedmap = RankMap.sortHashMapByValues(nbestMap_pattern, true);
		Set<Instance> keys = rankedmap.keySet();
		Iterator iter = keys.iterator();
		String bestPattern = null;
		if (iter.hasNext()) {
			Instance bestPatternInst = (Instance) iter.next();
			bestPattern= (String) bestPatternInst.getData();
		}
		
		HashMap<Instance, Double> nbestMap_tuple = getNbestMap(outputSummaryDir,
				corpusName, tupleCandidates);
		LinkedHashMap rankedmap_t = RankMap.sortHashMapByValues(nbestMap_tuple, true);
		HashMap<Instance, Double> nbestMap_tuple_N = new HashMap<Instance, Double>();
		Set<Instance> ks = rankedmap_t.keySet();
		Iterator iks = ks.iterator();
		double rank = 1.0;
		while(iks.hasNext()){
			Instance in = (Instance)iks.next();
			nbestMap_tuple_N.put(in, 1/(rank++));
		}
		
		HashMap<String, Double> tupleScoreMap = new HashMap<String, Double>();
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
		
//		HashMap<String, float[]> wordMap = fvGenerator.getWordMap();
		Metric metric = new NormalizedDotProductMetric();
		for(String tuple : tupleCandidates){
			double coverageScore = 0.0;
			String[] tokTup = tuple.split(" ");
			String[] tokPat = bestPattern.split(" ");
			for(String tokT : tokTup)
				for(String tokP : tokPat){
					tokP = cleaning(tokP.toLowerCase());
					float[] wordVector_P = wordMap.get(tokP);
					
					tokT = cleaning(tokT.toLowerCase());
					float[] wordVector_T = wordMap.get(tokT);
									
					if(wordVector_P != null && wordVector_T != null){
						int[] idx = new int[wordVector_P.length];
						for (int a = 0; a < wordVector_P.length; a++) {
							idx[a] = a;
						}
						double[] wordVector_T_D = new double[wordVector_T.length];
						double[] wordVector_P_D = new double[wordVector_P.length];
						for(int a = 0; a < wordVector_T.length; a++){
							wordVector_T_D[a] = wordVector_T[a];
							wordVector_P_D[a] = wordVector_P[a];
						}
							
						FeatureVector fv_t = new FeatureVector(idx, wordVector_T_D);
						FeatureVector fv_p = new FeatureVector(idx, wordVector_P_D);
	
						coverageScore += (1- metric.distance(fv_t, fv_p));
					}
					
				}
			
			double fluencyScore = 0.0;
			Set<Instance> set = nbestMap_tuple_N.keySet();
			Iterator iterKey = set.iterator();
			while(iterKey.hasNext()){
				Instance i = (Instance)iterKey.next();
				if(tuple.equals((String)i.getData())){
					fluencyScore = nbestMap_tuple_N.get(i);
					break;
				}
			}
			
			fluencyScore = 1/(1+Math.exp(-fluencyScore));
			double score = 0.7*coverageScore + 0.3*fluencyScore;
			tupleScoreMap.put(tuple, score);
		}

		HashMap sortedScores = RankMap.sortHashMapByValues(tupleScoreMap, false);
		String sent = null;
		Set<String> ts = sortedScores.keySet();
		Iterator its = ts.iterator();
		while(its.hasNext()){
			sent = (String)its.next();
		}
		return sent;
	}
	
	private ArrayList<String> generateSentenceByFusion(InstanceList cluster, String content){

		// Node Alignment
		SemanticGraph graph = new SemanticGraph();
		IndexedWord startNode = new IndexedWord();
		startNode.setIndex(-1);
		startNode.setDocID("-1");
		startNode.setSentIndex(-1);
		startNode.setLemma("START");
		startNode.setValue("START");
		startNode.setTag("START");
		graph.addRoot(startNode);

		IndexedWord endNode = new IndexedWord();
		endNode.setIndex(-2);
		endNode.setDocID("-2");
		endNode.setSentIndex(-2);
		endNode.setLemma("END");
		endNode.setValue("END");
		endNode.setTag("END");
		endNode.setOriginalText("END");
		
		if(content == "tuple"){
			for (int i = 0; i < cluster.size(); i++) {
				Instance inst = cluster.get(i);
				Pattern p = (Pattern) inst.getSource();
				Tuple t = p.getTuple();
				ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
				wordList.addAll(t.getArg1());
				wordList.addAll(t.getRel());
				wordList.addAll(t.getArg2());
				IndexedWord firstVertex = wordList.get(0);
				IndexedWord flag = getSimilarVertex(graph, firstVertex);
				if (flag == null) {
					graph.addEdge(startNode, firstVertex, null, 0.0, false);
				}

				for (int j = 0; j < wordList.size() - 1; j++) {
					IndexedWord source = wordList.get(j);
					IndexedWord flagSource = getSimilarVertex(graph, source);
					IndexedWord dest = wordList.get(j + 1);
					IndexedWord flagdest = getSimilarVertex(graph, dest);
					if (flagSource == null) {

						graph.addEdge(source, dest, null, 0.0, false);

					} else if (flagSource != null && flagdest == null) {

						graph.addEdge(flagSource, dest, null, 0.0, false);

					} else if (flagSource != null && flagdest != null) {
						SemanticGraphEdge edge = graph
								.getEdge(flagSource, flagdest);
						if (edge == null) {
							graph.addEdge(flagSource, flagdest, null, 0.0, false);
						}
					}
				}

				IndexedWord lastWord = wordList.get(wordList.size() - 1);
				IndexedWord flagLastWord = getSimilarVertex(graph, lastWord);
				IndexedWord flagEndRoot = getSimilarVertex(graph, endNode);

				if (flagLastWord != null && flagEndRoot == null) {
					graph.addEdge(flagLastWord, endNode, null, 0.0, false);
				} else if (flagLastWord != null && flagEndRoot != null) {
					SemanticGraphEdge edge = graph.getEdge(flagLastWord,
							flagEndRoot);
					if (edge == null) {
						graph.addEdge(flagLastWord, flagEndRoot, null, 0.0, false);

					}
				}
			}
		}else if(content == "pattern"){
			for (int i = 0; i < cluster.size(); i++) {
				Instance inst = cluster.get(i);
				Pattern p = (Pattern) inst.getSource();
				ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
				// replaced to pattern representation
				for (IndexedWord iw : p.getArg1()) {
					if (iw.equals(p.getArg1().getHead())) {
						iw.setOriginalText(p.getArg1().getHead().ner()
								.toUpperCase().replaceAll(" ", "_"));
					}
					wordList.add(iw);
				}
				wordList.addAll(p.getRel());
				for (IndexedWord iw : p.getArg2()) {
					if (iw.equals(p.getArg2().getHead())) {
						iw.setOriginalText(p.getArg2().getHead().ner()
								.toUpperCase().replaceAll(" ", "_"));
					}
					wordList.add(iw);
				}

				IndexedWord firstVertex = wordList.get(0);
				IndexedWord flag = getSimilarVertex(graph, firstVertex);
				if (flag == null) {
					graph.addEdge(startNode, firstVertex, null, 0.0, false);
				}

				for (int j = 0; j < wordList.size() - 1; j++) {
					IndexedWord source = wordList.get(j);
					IndexedWord flagSource = getSimilarVertex(graph, source);
					IndexedWord dest = wordList.get(j + 1);
					IndexedWord flagdest = getSimilarVertex(graph, dest);
					if (flagSource == null) {

						graph.addEdge(source, dest, null, 0.0, false);

					} else if (flagSource != null && flagdest == null) {

						graph.addEdge(flagSource, dest, null, 0.0, false);

					} else if (flagSource != null && flagdest != null) {
						SemanticGraphEdge edge = graph
								.getEdge(flagSource, flagdest);
						if (edge == null) {
							graph.addEdge(flagSource, flagdest, null, 0.0, false);
						}
					}
				}

				IndexedWord lastWord = wordList.get(wordList.size() - 1);
				IndexedWord flagLastWord = getSimilarVertex(graph, lastWord);
				IndexedWord flagEndRoot = getSimilarVertex(graph, endNode);

				if (flagLastWord != null && flagEndRoot == null) {
					graph.addEdge(flagLastWord, endNode, null, 0.0, false);
				} else if (flagLastWord != null && flagEndRoot != null) {
					SemanticGraphEdge edge = graph.getEdge(flagLastWord,
							flagEndRoot);
					if (edge == null) {
						graph.addEdge(flagLastWord, flagEndRoot, null, 0.0, false);
					}
				}
				
			}
		}

		
		
/*		ArrayList<ArrayList<IndexedWord>> paths = new ArrayList<ArrayList<IndexedWord>>();
		for(IndexedWord last : lastWords){
			List<IndexedWord> path = graph.getPathToRoot(last);
			path.add(0, last);
			ArrayList<IndexedWord> reversePath = new ArrayList<IndexedWord>();
			for(int i = path.size()-2; i>=0; i--){
				reversePath.add(path.get(i));
			}
			
			ArrayList<IndexedWord> reversePath = 
					(ArrayList<IndexedWord>) graph.getShortestDirectedPathNodes(startNode, last);
			reversePath.remove(0);
			paths.add(reversePath);
		}*/
			

//		System.out.println("Begin travel the graph to generate new tuples");
//		ArrayList<ArrayList<IndexedWord>> paths = travelAllPaths(graph, endNode);
		Method method;
		FutureTask task = null;
		try {
			method = getClass().getDeclaredMethod("travelAllPaths", 
					new Class[]{SemanticGraph.class, IndexedWord.class});
			List<Object> args = new ArrayList<Object>();
			args.add(graph);
			args.add(endNode);
			Callable call = new CallableTask(this, method, args);
			task = new FutureTask(call);
			Thread thread = new Thread(task);
			thread.setDaemon(true);
			thread.start();
		} catch (NoSuchMethodException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (SecurityException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}

		ArrayList<ArrayList<IndexedWord>> paths = null;
		try{
			
			paths = (ArrayList<ArrayList<IndexedWord>>) task.get(20, TimeUnit.SECONDS);
			
		}catch(Exception e){
			System.out.println("tuple fusion can't be fininshed in time");
			return null;
		}

		ArrayList<List<IndexedWord>> filteredPaths = new ArrayList<List<IndexedWord>>();
		HashSet<String> set = new HashSet<String>();
		for (int i = 0; i < paths.size(); i++) {
			List<IndexedWord> path = paths.get(i);
			StringBuilder sb = new StringBuilder();
			for (IndexedWord iw : path) {
				// keep consistent with dictionary
				sb.append(iw.originalText().replaceAll(" ", "_") + " ");
			}
			if (!set.contains(sb.toString().trim())) {
				filteredPaths.add(path);
				set.add(sb.toString().trim());
			}
		}

		ArrayList<String> merged = new ArrayList<String>();
		for (List<IndexedWord> path : filteredPaths) {
			StringBuilder sb = new StringBuilder();
			for (IndexedWord iw : path) {
				// keep consistent with dictionary
				sb.append(iw.originalText().replaceAll(" ", "_") + " ");
			}
			// prevent impossible lookup in dictionary
			if (sb.toString().trim().equals("")
					|| sb.toString().trim().equals(" "))
				continue;
			merged.add(sb.toString().trim());
		}

		return Merger.process(merged);
	}
	
	public ArrayList<ArrayList<IndexedWord>> travelAllPaths(
			SemanticGraph graph, IndexedWord endNode) {

		ArrayList<ArrayList<IndexedWord>> ret = new ArrayList<ArrayList<IndexedWord>>();

		Stack<IndexedWord> stack = new Stack<IndexedWord>();
		Stack<IndexedWord> path = new Stack<IndexedWord>();
		HashSet<IndexedWord> candidatePoints = new HashSet<IndexedWord>();

		// insert START
		stack.add(graph.getFirstRoot());
		boolean stackEmpty = false;
		boolean pathArriveEnd = false;

		while (!stack.isEmpty()) {

			if (!path.isEmpty() && stack.peek().index() == -2) {
				pathArriveEnd = true;
			}

			boolean containsCandidate = false;

			// if path arrive end
			if (pathArriveEnd) {

				ArrayList<IndexedWord> pa = new ArrayList<IndexedWord>();
				for (IndexedWord iw : path) {
					pa.add(iw);
				}
				ret.add(pa);

				// pop end
				stack.pop();
				if (stack.isEmpty()) {
					stackEmpty = true;
					break;
				}

				// Backtracking path, using top element of stack to decide
				// backtracking point.
				while (!path.isEmpty()) {

					if (stackEmpty == false) {
						boolean isSibing = graph.getSiblings(path.peek())
								.contains(stack.peek());
						int isParent = graph.isAncestor(stack.peek(),
								path.peek());

						// reach end
						if (stack.peek().index() == -2) {
							pathArriveEnd = true;
							break;
						} else
							pathArriveEnd = false;

						// case 1: if stack.peek is the child of path.peek. then
						// insert
						if (isParent == 1 && !path.contains(stack.peek())) {
							path.push(stack.peek());
							break;

							// case 2: if stack.peek is the sibling of
							// path.peek. then walk towards sibling.
						} else if (isSibing && !path.contains(stack.peek())) {
							if (path.size() == 1)
								break;

							path.pop();

							int parent = graph.isAncestor(stack.peek(),
									path.peek());
							if (parent == 1) {
								path.push(stack.peek());
								break;
							} else {
								// find the parent of stack.peek on path
								do {
									path.pop();

								} while (!path.empty()
										&& graph.isAncestor(stack.peek(),
												path.peek()) != 1);

								path.push(stack.peek());
								break;
							}

							// case 3: stack.peek() equals path.peek()
						} else if (stack.peek().equals(path.peek())) {

							if (path.size() == 1)
								break;

							IndexedWord flag = null;
							do {
								flag = path.pop();

							} while (!path.empty() && path.peek().index() != -1);

							if (path.empty())
								break;

							do {
								stack.pop();
								if (stack.isEmpty()) {
									stackEmpty = true;
									break;
								}

							} while (!path.empty()
									&& graph.isAncestor(stack.peek(),
											path.peek()) != 1
									|| flag.equals(stack.peek()));

							if (stackEmpty == false) {
								if (!candidatePoints.contains(stack.peek())) {
									candidatePoints.add(stack.peek());
									path.push(stack.peek());
								} else {
									containsCandidate = true;
									// choose candidate stack.peek
									do {
										stack.pop();
										if (stack.isEmpty()) {
											stackEmpty = true;
											break;
										}

									} while (!path.empty()
											&& graph.isAncestor(stack.peek(),
													path.peek()) != 1
											|| flag.equals(stack.peek())
											|| candidatePoints.contains(stack
													.peek()));
								}

							}

							break;

						} else if (graph.isAncestor(endNode, path.peek()) == 1) {
							/*ArrayList<IndexedWord> lastP = ret.get(ret.size()-1);
							IndexedWord lastt = lastP.get(lastP.size()-1);
							IndexedWord lastpath = path.peek();
							if(!lastt.equals(lastpath)){
								 pathArriveEnd = true;
							}
							 break;*/
							pathArriveEnd = true;
							break;
							
						} else
							path.pop();
								
					} else
						break;
				}

			} else {

				if (!path.contains(stack.peek()))
					path.push(stack.peek());
				else {
					// choose another way
					stack.pop();
					if (stack.isEmpty())
						stackEmpty = true;

					// Backtracking path, using top element of stack to decide
					// backtracking point.
					while (!path.isEmpty()) {

						if (stackEmpty == false) {
							boolean isSibing = graph.getSiblings(path.peek())
									.contains(stack.peek());
							int isParent = graph.isAncestor(stack.peek(),
									path.peek());

							// reach end
							if (stack.peek().index() == -2) {
								pathArriveEnd = true;
								break;
							} else
								pathArriveEnd = false;

							// case 1: if stack.peek is the child of path.peek.
							// then insert
							if (isParent == 1 && !path.contains(stack.peek())) {
								path.push(stack.peek());
								break;

								// case 2: if stack.peek is the sibling of
								// path.peek. then walk towards sibling.
							} else if (isSibing && !path.contains(stack.peek())) {
								if (path.size() == 1)
									break;

								path.pop();
								int parent = graph.isAncestor(stack.peek(),
										path.peek());
								if (parent == 1) {
									path.push(stack.peek());
									break;
								} else {
									// find the parent of stack.peek on path
									do {
										path.pop();

									} while (graph.isAncestor(stack.peek(),
											path.peek()) != 1);

									path.push(stack.peek());
									break;
								}

								// case 3: stack.peek() equals path.peek()
							} else if (stack.peek().equals(path.peek())) {

								if (path.size() == 1)
									break;

								// flag is the next token of start, clear path
								IndexedWord flag = null;
								boolean jump = false;
								do {
									if (graph.isAncestor(endNode, path.peek()) == 1) {
										jump = true;
										break;
									}
									flag = path.pop();

								} while (path.peek().index() != -1);

								if (jump == true) {
									pathArriveEnd = true;
									break;
								}

								do {
									stack.pop();
									if (stack.isEmpty()) {
										stackEmpty = true;
										break;
									}

								} while (graph.isAncestor(stack.peek(),
										path.peek()) != 1
										|| flag.equals(stack.peek()));

								if (stackEmpty == false) {
									if (!candidatePoints.contains(stack.peek())) {
										candidatePoints.add(stack.peek());
										path.push(stack.peek());
									} else {
										containsCandidate = true;
										// choose another candidate stack.peek
										do {
											stack.pop();
											if (stack.isEmpty()) {
												stackEmpty = true;
												break;
											}

										} while (graph.isAncestor(stack.peek(),
												path.peek()) != 1
												|| flag.equals(stack.peek())
												|| candidatePoints
														.contains(stack.peek()));
									}

								}

								break;

							} else if (graph.isAncestor(endNode, path.peek()) == 1) {
								/*ArrayList<IndexedWord> lastP = ret.get(ret.size()-1);
								IndexedWord lastt = lastP.get(lastP.size()-1);
								IndexedWord lastpath = path.peek();
								if(!lastt.equals(lastpath)){
									 pathArriveEnd = true;
								}
								 break;*/
								pathArriveEnd = true;
								break;
							}else
								path.pop();
						} else
							break;
					}
				}
			}

			if (pathArriveEnd == true)
				continue;

			if (containsCandidate == true)
				continue;

			if (stackEmpty == true)
				break;

			Iterable<SemanticGraphEdge> iter = graph.outgoingEdgeIterable(stack
					.pop());
			for (SemanticGraphEdge edge : iter)
				stack.push(edge.getDependent());
		}

		stack.clear();
		path.clear();
		candidatePoints.clear();
		return ret;
	}
	
	private HashMap<Instance, Double> getNbestMap(String outputSummaryDir,
			String corpusName, ArrayList<String> candidates){

		PrintWriter nbest = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".nbest");

		for (String s : candidates) {
			nbest.println(s);
		}
		nbest.close();

		String[] cmd = {
				"/home/peng/Develop/Workspace/Mavericks/platform/src"
						+ "/main/java/edu/pengli/nlp/platform/algorithms/neuralnetwork/RNNLM/rnnlm",
				"-rnnlm", outputSummaryDir + "/" + corpusName + ".rnnlm.model",
				"-test", outputSummaryDir + "/" + corpusName + ".nbest",
				"-nbest", "-debug", "0" };

		ProcessBuilder builder = new ProcessBuilder(cmd);
		builder.redirectOutput(new File(outputSummaryDir + "/" + corpusName
				+ ".scores"));
		Process proc;
		try {
			proc = builder.start();
			while (proc.waitFor() != 0)
				TimeWait.waiting(100);
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}catch (InterruptedException e) {
			e.printStackTrace();
		}

		BufferedReader in_score = FileOperation.getBufferedReader(new File(
				outputSummaryDir), corpusName + ".scores");
		BufferedReader in_nbest = FileOperation.getBufferedReader(new File(
				outputSummaryDir), corpusName + ".nbest");
		String input_nbest, input_score;
		HashMap<Instance, Double> nbestmap = new HashMap<Instance, Double>();
		int i = 0;
		try {
			while ((input_nbest = in_nbest.readLine()) != null
					&& (input_score = in_score.readLine()) != null) {
				Instance inst = new Instance(candidates.get(i++), null, null,
						input_nbest);
				if (input_score.equals("-inf")) {
					nbestmap.put(inst, -100.0);
				} else {
					nbestmap.put(inst, Double.valueOf(input_score));
				}
			}
			
			in_score.close();
			in_nbest.close();
		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		return nbestmap;

	}
	
	private IndexedWord getSimilarVertex(SemanticGraph graph, IndexedWord vertex) {

		// the word are appear in the same tuple that have the same docId,
		// sentId, and index
		if (graph.containsVertex(vertex))
			return vertex;
		else {
			// the word are appear in different tuple that have the same
			// mention.
			String pattern = null;
			if (vertex.tag().startsWith("PRP") || vertex.tag().startsWith("DT")) { // his
																					// he
																					// should
																					// be
																					// not
																					// alignment.
																					// his/PRP$
				pattern = "^" + vertex.originalText();
			} else
				pattern = "^" + vertex.lemma();

			List<IndexedWord> similarWords = graph
					.getAllNodesByWordPattern(pattern);
			if (similarWords.isEmpty())
				return null;
			else {
				IndexedWord iw = similarWords.get(0);
				if (iw.originalText().equals("the")
						|| iw.originalText().equals("to")
						|| iw.originalText().equals("of")
						|| iw.originalText().equals("have"))
					return null;
				if (iw.docID().equals(vertex.docID())
						&& iw.sentIndex() == vertex.sentIndex())
					return null;
				else {
					return iw;
				}

			}
		}
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
	
	private String buildStrVector(int m, int n, int i, int j,
			Clustering clusters, int instanceSize, Metric metric) {

		InstanceList[] cs = clusters.getClusters();

		InstanceList clusteri = cs[i];
		Instance instm = clusteri.get(m);

		InstanceList clusterj = cs[j];
		Instance instn = clusterj.get(n);

		double sim = 1- metric.distance((FeatureVector)instm.getData(), 
				(FeatureVector)instn.getData());

		int mdirect = 0;

		for (int k = 0; k < clusters.getNumClusters(); k++) {
			if (k == i) {
				mdirect += m;
				break;
			} else
				mdirect += cs[k].size();
		}

		int ndirect = 0;
		for (int k = 0; k < clusters.getNumClusters(); k++) {
			if (k == j) {
				ndirect += n;
				break;
			} else
				ndirect += cs[k].size();
		}

		StringBuffer sb = new StringBuffer();
		for (int k = 0; k < instanceSize; k++) {
			if (k == mdirect || k == ndirect) {
				sb.append(sim + " ");
			} else
				sb.append(0 + " ");
		}

		return sb.toString().trim();
	}

}
