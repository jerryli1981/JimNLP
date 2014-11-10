package edu.pengli.nlp.conference.acl2015.generation;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Stack;

import edu.pengli.nlp.conference.acl2015.pipe.FeatureVectorGenerator;
import edu.pengli.nlp.conference.acl2015.pipe.FramenetTagger;
import edu.pengli.nlp.conference.acl2015.pipe.HeadAnnotation;
import edu.pengli.nlp.conference.acl2015.pipe.FeatureVectorGenerator.WordEntry;
import edu.pengli.nlp.conference.acl2015.types.Argument;
import edu.pengli.nlp.conference.acl2015.types.Category;
import edu.pengli.nlp.conference.acl2015.types.InformationItem;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Predicate;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.algorithms.classify.Clustering;
import edu.pengli.nlp.platform.algorithms.classify.KMeans;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;
import edu.pengli.nlp.platform.types.NormalizedDotProductMetric;
import edu.pengli.nlp.platform.types.SparseVector;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.TimeWait;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.util.CoreMap;
import simplenlg.framework.NLGFactory;
import simplenlg.lexicon.Lexicon;
import simplenlg.phrasespec.NPPhraseSpec;
import simplenlg.phrasespec.PPPhraseSpec;
import simplenlg.phrasespec.SPhraseSpec;
import simplenlg.phrasespec.VPPhraseSpec;
import simplenlg.realiser.english.Realiser;

public class AbstractiveGeneration {

	NLGFactory nlgFactory;
	Realiser realiser;

	static FramenetTagger framenetTagger;

	static FeatureVectorGenerator fvGenerator;

	public AbstractiveGeneration() {
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);
	}

	private String realization(Pattern p, SemanticGraph graph) {

		Tuple t = p.getTuple();
		SPhraseSpec newSent = nlgFactory.createClause();

		IndexedWord arg1iw = t.getArg1().getHead();
		NPPhraseSpec subjectNp = generateNP(graph, arg1iw);
		newSent.setSubject(subjectNp);

		VPPhraseSpec vp = generateVP(graph, t.getRel(), t.getArg2());
		newSent.setVerbPhrase(vp);

		String output = realiser.realiseSentence(newSent);

		return output;

	}

	// search the tree recursively
	private IndexedWord searchObjforPrep(SemanticGraph graph,
			IndexedWord prepNode) {

		IndexedWord obj = null;
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2];
		int headIdx = prepNode.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		boolean stop = false;
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph
					.outgoingEdgeIterable(prepNode);

			for (SemanticGraphEdge edge : iter) {
				GrammaticalRelation dgr = edge.getRelation();
				if (dgr.toString().endsWith("obj")
						|| dgr.toString().endsWith("pcomp")) {
					obj = edge.getDependent();
					stop = true;
				}
				int depIdx = edge.getDependent().index();
				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
			if (stop == true)
				break;
		}

		return obj;
	}

	private NPPhraseSpec generateNP(SemanticGraph graph, IndexedWord head) {

		NPPhraseSpec np = nlgFactory.createNounPhrase();
		np.setHead(head.originalText());
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2];
		int headIdx = head.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph.outgoingEdgeIterable(graph
					.getNodeByIndex(s));
			for (SemanticGraphEdge edge : iter) {
				if (edge.getGovernor().index() == edge.getDependent().index())
					continue; // prevent infitive recusion

				GrammaticalRelation gr = edge.getRelation();

				int depIdx = edge.getDependent().index();

				if (gr.toString().equals("prep")) {
					
					IndexedWord prep = edge.getDependent();
					IndexedWord obj = searchObjforPrep(graph,
							edge.getDependent());
/*					if (obj != null) {
						PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(ppp);
						} else
							np.setPostModifier(ppp);
					}*/

					continue; // do not deep travel any more

				} else if (gr.toString().equals("nn")) {
					NPPhraseSpec nounModifier = generateNP(graph,
							edge.getDependent());
					if (edge.getDependent().index() < head.index()) {
						if (np.getPreModifiers().size() != 0) {
							np.addPreModifier(nounModifier);
						} else
							np.setPreModifier(nounModifier);
					} else {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(nounModifier);
						} else
							np.setPostModifier(nounModifier);
					}

					continue;

				} else if (gr.toString().equals("conj")) {

					Iterable<SemanticGraphEdge> children = graph
							.outgoingEdgeIterable(edge.getGovernor());
					IndexedWord cc = null;
					for (SemanticGraphEdge child : children) {
						GrammaticalRelation dgr = child.getRelation();
						if (dgr.toString().equals("cc")) {
							cc = child.getDependent();
						}
					}
					NPPhraseSpec nounModifier = generateNP(graph,
							edge.getDependent());
					if (cc != null) {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(cc.originalText());
							np.addPostModifier(nounModifier);
						} else {
							np.setPostModifier(cc.originalText());
							np.addPostModifier(nounModifier);
						}
					} else {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(nounModifier);
						} else {
							np.addPostModifier(nounModifier);
						}
					}

					continue;
				} else if (gr.toString().equals("det")
						|| gr.toString().equals("poss")||
						gr.toString().equals("neg")) {
					IndexedWord det = edge.getDependent();
					np.setSpecifier(det.value());
				} else if (gr.toString().equals("num")) {

					IndexedWord numModifier = edge.getDependent();
					np.setSpecifier(numModifier.value());

				} else if (gr.toString().equals("amod")) {
					IndexedWord adjMod = edge.getDependent();
					if (adjMod.index() < head.index()) {
						if (np.getPreModifiers().size() != 0) {
							np.addPreModifier(adjMod.originalText());
						} else
							np.setPreModifier(adjMod.originalText());

					} else {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(adjMod.originalText());
						} else
							np.setPostModifier(adjMod.originalText());
					}

				} else
					continue; // this is ignore all the other children

				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}// end of typed Dependency
		}

		return np;
	}

	private VPPhraseSpec generateVP(SemanticGraph graph, Predicate pred,
			Argument arg2) {
	
		VPPhraseSpec vp = nlgFactory.createVerbPhrase();
		
		//here is the whole pred mention as vp head, but why? keep readability?
		vp.setHead(pred.originaltext());
		
/*		// set aux of the headVerb
		Iterable<SemanticGraphEdge> children = graph
				.outgoingEdgeIterable(headVp);
		for (SemanticGraphEdge edge : children) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("aux")) {
				vp.setPreModifier(edge.getDependent().originalText());
				break;
			}
		}*/
		
		//using the shortest dependency path between headVp and head Argument 
		//to decide generation type
		
		//already test whether headVp is null when pattern generation. so 
		//here do not need to test
		IndexedWord headVp = pred.getHead();
		IndexedWord argHead = arg2.getHead();
		ArrayList<IndexedWord> toks = new ArrayList<IndexedWord>();
		for(int i = headVp.index(); i< arg2.get(arg2.size()-1).index(); i++){
			IndexedWord word = graph.getNodeByIndexSafe(i);
			if(word != null)
				toks.add(word);
		}
		
		List<SemanticGraphEdge> path = graph.getShortestDirectedPathEdges(headVp, argHead);
/*		List<SemanticGraphEdge> path = new ArrayList<SemanticGraphEdge>();
		for (int i = 0; i < toks.size(); i++) {
			for (int j = 0; j < toks.size(); j++) {
				if (i == j)
					continue;
				else {
					IndexedWord ai = toks.get(i);
					IndexedWord aj = toks.get(j);
					if (ai == null || aj == null)
						continue;
					List<SemanticGraphEdge> edge = graph.getAllEdges(ai, aj);
					path.addAll(edge);
				}
			}
		}*/

		if(path == null)
		{
			vp.setObject(arg2.toString());
			return vp;
		}

		IndexedWord prep = null;
		IndexedWord dobj = null;
		IndexedWord pobj = null;
		for (SemanticGraphEdge edge : path) {
			GrammaticalRelation dgr = edge.getRelation();
			if (dgr.toString().equals("pobj")){
				prep = edge.getGovernor();
				pobj = edge.getDependent();				
				PPPhraseSpec ppp = generatePrepP(graph, prep, pobj);
				vp.addPostModifier(ppp);
				
			}else if(dgr.toString().equals("dobj")){
				dobj = edge.getDependent();
				
				NPPhraseSpec dirObjNp = generateNP(graph, dobj);
				vp.setObject(dirObjNp);
			}else if(dgr.toString().endsWith("comp")){
				NPPhraseSpec np = generateNP(graph, edge.getDependent());
				vp.addComplement(np);
			}else if(dgr.toString().endsWith("mod") && edge.getGovernor().equals(headVp)){
				NPPhraseSpec np = generateNP(graph, edge.getDependent());
				vp.addPostModifier(np);
			}
			
		}
		
		return vp;
	}

	private PPPhraseSpec generatePrepP(SemanticGraph graph, IndexedWord prep,
			IndexedWord np) {
		PPPhraseSpec ppp = nlgFactory.createPrepositionPhrase();
		ppp.setPreposition(prep.originalText());
		NPPhraseSpec npp = generateNP(graph, np);
		ppp.setObject(npp);
		return ppp;
	}

	private HashMap<String, Double> getNbestMap(String outputSummaryDir,
			String corpusName, InstanceList corpus)
			throws NumberFormatException, IOException {

		// tuple filtering
		PrintWriter nbest = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".nbest");
		for (Instance doc : corpus) {
			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();
			for (CoreMap sent : map.keySet()) {

				ArrayList<Tuple> tuples = map.get(sent);
				for (Tuple t : tuples) {
					nbest.println(t.getRel().toString());
				}
			}

		}

		nbest.close();

		String[] cmd = { "/home/peng/Develop/Workspace/rnnlm-0.4b/rnnlm",
				"-rnnlm", "/home/peng/Develop/Workspace/rnnlm-0.4b/model_2",
				"-test", outputSummaryDir + "/" + corpusName + ".nbest",
				"-nbest", "-debug", "0" };

		ProcessBuilder builder = new ProcessBuilder(cmd);
		builder.redirectOutput(new File(outputSummaryDir + "/" + corpusName
				+ ".scores"));
		Process proc = builder.start();

		try {
			while (proc.waitFor() != 0)
				TimeWait.waiting(100);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}

		BufferedReader in_score = FileOperation.getBufferedReader(new File(
				outputSummaryDir), corpusName + ".scores");
		BufferedReader in_nbest = FileOperation.getBufferedReader(new File(
				outputSummaryDir), corpusName + ".nbest");
		String input_nbest, input_score;
		HashMap<String, Double> nbestmap = new HashMap<String, Double>();
		while ((input_nbest = in_nbest.readLine()) != null
				&& (input_score = in_score.readLine()) != null) {
			if (input_score.equals("-inf")) {
				nbestmap.put(input_nbest, -100.0);
			} else
				nbestmap.put(input_nbest, Double.valueOf(input_score));

		}
		in_score.close();
		in_nbest.close();
		return nbestmap;

	}

	private void generatePatterns(String outputSummaryDir, String corpusName,
			InstanceList corpus, HeadAnnotation headAnnotator) throws Exception {

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".ser"));

		corpus.readObject(in);
		in.close();

		HashMap<String, Double> nbestmap = getNbestMap(outputSummaryDir,
				corpusName, corpus);

		HashSet<Pattern> patternSet = new HashSet<Pattern>();

		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" + corpusName + ".patterns"));
		PrintWriter outt = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".tuples");
		
		double threshould = -20.0;
		for (Instance doc : corpus) {
			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();
			for (CoreMap sent : map.keySet()) {
				outt.println(sent.toString());
				ArrayList<Tuple> tuples = map.get(sent);
				for (Tuple t : tuples) {
					
					if (t.getRel().toString().equals("say"))
						continue;	
					double score = nbestmap.get(t.getRel().toString());
					if (score < threshould)
						continue;
					
					outt.println(t.toString());
					edu.pengli.nlp.conference.acl2015.types.Argument arg1 = headAnnotator
							.annotateArgHead(t.getArg1(), sent);
					t.setArg1(arg1);

					edu.pengli.nlp.conference.acl2015.types.Argument arg2 = headAnnotator
							.annotateArgHead(t.getArg2(), sent);
					t.setArg2(arg2);
					
					//for later sentence realization to get head verb
					edu.pengli.nlp.conference.acl2015.types.Predicate pre = headAnnotator
							.annotatePredicateHead(t.getRel(), sent);
					t.setRel(pre);
					
					//for complicated arguments, just ignore, so arg may be null
					if(arg1 == null || arg2 == null)
						continue;
					
					//tuple with no head should not go into clustering ?
					if(pre.getHead() == null)
						continue;

					if (arg1.getHead() != null && arg2.getHead() != null) {

						framenetTagger.annotate(arg1, arg2, t);

						if (!arg1.getHead().ner().equals("O")
								&& !arg2.getHead().ner().equals("O")) {

							//here is ok, just for clustering,
							//when realization we use tuple in pattern. 
							Pattern p = new Pattern(arg1.getHead().ner(), t
									.getRel().toString(),
									arg2.getHead().ner(), sent, t);
							patternSet.add(p);
						}
					}
				}
			}
		}

		outt.close();
		out.writeObject(patternSet);
		out.close();
	}

	private InstanceList featureEngineering(HashSet<Pattern> patternSet) {

		InstanceList instances = new InstanceList(new Noop());

		for (Pattern p : patternSet) {
			FeatureVector fv = fvGenerator.getFeatureVector(p);
			Instance inst = new Instance(fv, null, null, p);
			instances.add(inst);
		}

		return instances;
	}

	private InstanceList featureEngineeringOnCategory(String categoryId) {

		InstanceList seeds = new InstanceList(new Noop());
		Category[] cats = Category.values();
		for (Category cat : cats) {
			if (cat.getId() == Integer.parseInt(categoryId)) {
				Map<String, String[]> aspects = cat.getAspects(cat.getId());
				Set<String> keys = aspects.keySet();
				for (String key : keys) {
					String[] keywords = aspects.get(key);
					FeatureVector fv = fvGenerator.getFeatureVector(keywords);
					Instance inst = new Instance(fv, null, null, key);
					seeds.add(inst);
				}
			}
		}

		return seeds;
	}

	private void generateFinalSummary(String outputSummaryDir,
			String corpusName, Clustering predicted, InstanceList seeds) {
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);
		HashSet<InstanceList> set = new HashSet<InstanceList>();
		for (Instance seed : seeds) {
			FeatureVector seedFv = (FeatureVector) seed.getData();
			InstanceList[] clusters = predicted.getClusters();
			float Max = Float.MIN_VALUE;
			InstanceList bestCluster = null;
			for (InstanceList cluster : clusters) {
				SparseVector meanVec = KMeans.mean(cluster);
				float dist = 0;
				for (int i = 0; i < meanVec.getIndices().length; i++) {
					dist += seedFv.getValues()[i] * meanVec.getValues()[i];
				}
				if (dist >= Max && !set.contains(cluster)) {
					Max = dist;
					bestCluster = cluster;

				}
			}
			set.add(bestCluster);
			for (int i = 0; i < bestCluster.size(); i++) {
				Instance inst = bestCluster.get(i);
				Pattern p = (Pattern) inst.getSource();

				CoreMap annotation = p.getCoreMap();
				if(annotation.toString().contains("Two relatives of the man"
						+ " who attacked Amish school girls said Monday they were not molested by him 20 years ago"))
					System.out.println();
				SemanticGraph graph = annotation
						.get(BasicDependenciesAnnotation.class);
				String summarySent = realization(p, graph);
				System.out.println(p.getCoreMap().toString());
				System.out.println(p.getTuple().toString());
				System.out.println(p);
				System.out.println(summarySent);
				System.out.println();
				if (summarySent == null)
					continue;
				// String summarySent = p.getCoreMap().toString();
				// out.println(p.toString());
				// out.println(summarySent);
				// if(i > 1)break;
			}
			System.out.println();
			out.println();
		}

		out.close();
	}

	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, String categoryId)
			throws Exception {

		InstanceList docs = new InstanceList(pipeLine);

		
/*		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName); docs.addThruPipe(fIter);
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" +corpusName + ".ser")); docs.writeObject(out);	
		out.close();*/
		
/*		System.out.println("Begin generate patterns"); 
		HeadAnnotation headAnnotator = new HeadAnnotation(); 
		if(framenetTagger == null)
			framenetTagger = new FramenetTagger();
		generatePatterns(outputSummaryDir, corpusName, docs, headAnnotator );*/


		System.out.println("Begin summary generation");
		if (fvGenerator == null)
			fvGenerator = new FeatureVectorGenerator();
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".patterns"));
		HashSet<Pattern> patternSet = (HashSet<Pattern>) in.readObject();
		in.close();
		InstanceList instances = featureEngineering(patternSet);
		InstanceList seeds = featureEngineeringOnCategory(categoryId);
		int numClusters = 20;
		Metric metric = new NormalizedDotProductMetric();
		KMeans kmeans = new KMeans(new Noop(), numClusters, metric);
		Clustering predicted = kmeans.cluster(instances);
		generateFinalSummary(outputSummaryDir, corpusName, predicted, seeds);
	}

}
