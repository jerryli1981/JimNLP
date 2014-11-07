package edu.pengli.nlp.conference.acl2015.generation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
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
import edu.pengli.nlp.conference.acl2015.pipe.HeadExtractor;
import edu.pengli.nlp.conference.acl2015.pipe.FeatureVectorGenerator.WordEntry;
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

//	DBpediaTagger dbpediaTagger;

//	FreebaseTagger freebaseTagger;

//	WordnetTagger wordnetTagger;
	
	static FramenetTagger framenetTagger;
	
	static FeatureVectorGenerator fvGenerator;
	
	public AbstractiveGeneration(){
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);
	}

	/*
	 * current implementation just cove direct object and prep object, subject
	 * and predicate are necessary. object could be empty.
	 */

	private ArrayList<InformationItem> extractInformationItems(
			SemanticGraph graph) {

		HashSet<IndexedWord> predicates = new HashSet<IndexedWord>();
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2]; // index count from 1,
															// also contains
															// punc.
		int rootIdx = graph.getFirstRoot().index();
		marked[rootIdx] = true;
		stack.add(rootIdx);
		List<IndexedWord> sentenceSubjects = new ArrayList<IndexedWord>();
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph.outgoingEdgeIterable(graph
					.getNodeByIndex(s));
			for (SemanticGraphEdge edge : iter) {
				GrammaticalRelation gr = edge.getRelation();
				IndexedWord gov = edge.getGovernor();
				if (gr.toString().equals("nsubj")
						|| gr.toString().equals("dobj")
						|| (gr.toString().equals("prep") && gov.tag()
								.startsWith("VB"))) {
					predicates.add(edge.getGovernor());
				}

				// find sentence subject
				if (gr.toString().equals("nsubj") && gov.tag().startsWith("VB")) {
					Collection<IndexedWord> sibs = graph.getSiblings(edge
							.getDependent());
					for (IndexedWord sib : sibs) {
						GrammaticalRelation dgr = graph.reln(gov, sib);
						if (dgr.toString().equals("dobj")
								|| (dgr.toString().equals("prep"))) {

							sentenceSubjects.add(edge.getDependent());
						}
					}
				}

				int depIdx = edge.getDependent().index();
				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}

		ArrayList<InformationItem> possibleItems = new ArrayList<InformationItem>();

		if (sentenceSubjects.size() == 0)
			return possibleItems;

		for (IndexedWord p : predicates) {

			boolean subjectExist = false;
			boolean directObjectExist = false;
			boolean prepObjectExist = false;
			IndexedWord subject = null;
			IndexedWord directObject = null;
			IndexedWord prep = null;
			IndexedWord prepObject = null;

			// travel the graph

			stack = new Stack<Integer>();
			marked = new boolean[graph.size() * 2]; // index count from 1, also
													// contains punc.
			rootIdx = graph.getFirstRoot().index();
			marked[rootIdx] = true;
			stack.add(rootIdx);
			while (!stack.isEmpty()) {
				int s = stack.pop();
				Iterable<SemanticGraphEdge> iter = graph
						.outgoingEdgeIterable(p);
				for (SemanticGraphEdge edge : iter) {
					GrammaticalRelation gr = edge.getRelation();
					IndexedWord gov = edge.getGovernor();
					if (gr.toString().equals("nsubj")) {
						subjectExist = true;
						subject = edge.getDependent();
					}

					if (gr.toString().equals("dobj")) {
						directObjectExist = true;
						directObject = edge.getDependent();
					}

					if (gr.toString().equals("prep")
							&& gov.tag().startsWith("VB")) {

						Iterable<SemanticGraphEdge> children = graph
								.outgoingEdgeIterable(edge.getDependent());
						for (SemanticGraphEdge child : children) {
							GrammaticalRelation dgr = child.getRelation();
							if (dgr.toString().equals("pobj")) {
								prepObjectExist = true;
								prep = edge.getDependent();
								prepObject = child.getDependent();
							}
						}

					}

					int depIdx = edge.getDependent().index();
					if (!marked[depIdx]) {
						marked[depIdx] = true;
						stack.add(depIdx);
					}
				}
			}

			if (subjectExist == false && directObjectExist == true
					&& prepObjectExist == false) {
				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(directObject);
				possibleItems.add(new InformationItem(sentenceSubjects.get(0),
						p, obj));

			} else if (subjectExist == false && directObjectExist == false
					&& prepObjectExist == true) {

				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(sentenceSubjects.get(0),
						p, obj));

			} else if (subjectExist == true && directObjectExist == false
					&& prepObjectExist == false) {

				possibleItems.add(new InformationItem(subject, p, null));

			} else if (subjectExist == true && directObjectExist == true
					&& prepObjectExist == false) {
				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(directObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			} else if (subjectExist == true && directObjectExist == false
					&& prepObjectExist == true) {

				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			} else if (subjectExist == true && directObjectExist == true
					&& prepObjectExist == true) {
				// One Amish man craned his head out a buggy window
				ArrayList<IndexedWord> obj = new ArrayList<IndexedWord>();
				obj.add(directObject);
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			}
		}

		return possibleItems;

	}

	private String realization(Pattern p, SemanticGraph graph) {
		
        Tuple t = p.getTuple();
		SPhraseSpec newSent = nlgFactory.createClause();

		int arg1CoreLabelIdx = t.getArg1().get(0).index();
		IndexedWord arg1iw = graph.getNodeByIndexSafe(arg1CoreLabelIdx);
		
		int arg2CoreLabelIdx = t.getArg2().get(0).index();
		IndexedWord arg2iw = graph.getNodeByIndexSafe(arg2CoreLabelIdx);
		ArrayList<IndexedWord> objects = new ArrayList<IndexedWord>();
		objects.add(arg2iw);

		NPPhraseSpec subjectNp = generateNP(graph, arg1iw);

		newSent.setSubject(subjectNp);

		Predicate pred = t.getRel();
		IndexedWord headVp = null;
		
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.size() * 2]; 
		int rootIdx = graph.getFirstRoot().index();
		marked[rootIdx] = true;
		stack.add(rootIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<SemanticGraphEdge> iter = graph.outgoingEdgeIterable(graph
					.getNodeByIndex(s));
			for (SemanticGraphEdge edge : iter) {
				GrammaticalRelation gr = edge.getRelation();
				IndexedWord gov = edge.getGovernor();
				if (gr.toString().equals("nsubj")
						|| gr.toString().equals("dobj")
						|| (gr.toString().equals("prep") && gov.tag()
								.startsWith("VB"))) {
					
					for(CoreLabel tok : pred){
						int preCoreLabelIdx = tok.index();
						IndexedWord preiw = graph.getNodeByIndexSafe(preCoreLabelIdx);
						if(preiw == null)
							continue;
						if(preiw.equals(edge.getGovernor())){
							headVp = preiw;
						}
					}
				}

				int depIdx = edge.getDependent().index();
				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}
		
		if(headVp == null){
			for(CoreLabel cl : pred){
				if(cl.tag().startsWith("VB")){
					headVp = graph.getNodeByIndexSafe(cl.index());
				}
			}
		}
		if(headVp == null){
//			System.out.println("wired");
//			System.out.println(p.getCoreMap().toString());
//			System.exit(0);
			return null;
		}
		
		VPPhraseSpec vp = generateVP(graph, headVp, objects);

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
					String prep = edge.getDependent().originalText();
					IndexedWord obj = searchObjforPrep(graph,
							edge.getDependent());
					if (obj != null) {
						PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(ppp);
						} else
							np.setPostModifier(ppp);
					}

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
						|| gr.toString().equals("poss")) {
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

	private VPPhraseSpec generateVP(SemanticGraph graph, IndexedWord headVp,
			ArrayList<IndexedWord> object) {

		VPPhraseSpec vp = nlgFactory.createVerbPhrase();
		vp.setHead(headVp.originalText());
		// set aux of the headVerb
		Iterable<SemanticGraphEdge> children = graph
				.outgoingEdgeIterable(headVp);
		for (SemanticGraphEdge edge : children) {
			GrammaticalRelation gr = edge.getRelation();
			if (gr.toString().equals("aux")) {
				vp.setPreModifier(edge.getDependent().originalText());
				break;
			}
		}

		// set object
		if (object != null) {

			if (object.size() == 1) {
				// set direct object
				NPPhraseSpec dirObjNp = generateNP(graph, object.get(0));
				vp.setObject(dirObjNp);
			}

			if (object.size() == 2) {
				// set prep object from direct children
				String prep = object.get(0).originalText();
				IndexedWord obj = searchObjforPrep(graph, object.get(0));
				PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
				vp.setObject(ppp);

			}

			if (object.size() == 3) {

				// set direct and prep object
				NPPhraseSpec dirObjNp = generateNP(graph, object.get(0));
				vp.setObject(dirObjNp);

				String prep = object.get(1).originalText();
				IndexedWord obj = searchObjforPrep(graph, object.get(1));
				PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
				vp.setPostModifier(ppp);

			}

		}
		return vp;
	}

	private PPPhraseSpec generatePrepP(SemanticGraph graph, String prep,
			IndexedWord np) {
		PPPhraseSpec ppp = nlgFactory.createPrepositionPhrase();
		ppp.setPreposition(prep);
		NPPhraseSpec npp = generateNP(graph, np);
		ppp.setObject(npp);
		return ppp;
	}

	private void generatePatterns(String outputSummaryDir, String corpusName,
			InstanceList corpus, HeadExtractor headExtractor)
			throws Exception {

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".ser"));

		corpus.readObject(in);
		
		in.close();

		ObjectOutputStream  out = new ObjectOutputStream(new 
				FileOutputStream(outputSummaryDir + "/" + corpusName + ".patterns"));
		HashSet<Pattern> patternSet = new HashSet<Pattern>();
		for (Instance doc : corpus) {
			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();
			for (CoreMap sent : map.keySet()) {
								
				ArrayList<Tuple> tuples = map.get(sent);
				for (Tuple t : tuples) {
					if (t.getRel().toString().equals("said"))
						continue;
					
					edu.pengli.nlp.conference.acl2015.types.Argument arg1Head = headExtractor
							.extract(t.getArg1(), sent);

					
					edu.pengli.nlp.conference.acl2015.types.Argument arg2Head = headExtractor
							.extract(t.getArg2(), sent);
				

					if (arg1Head != null && arg2Head != null) {
						
						framenetTagger.annotate(arg1Head.get(0), arg2Head.get(0), t);	
						
						Tuple tmpTuple = new Tuple(arg1Head, t.getRel(), arg2Head);
						
						if(!arg1Head.get(0).ner().equals("O") 
								&& !arg2Head.get(0).ner().equals("O")){
							
							Pattern p = new Pattern(arg1Head.get(0).ner(), 
									t.getRel().toString(), arg2Head.get(0).ner(), sent, tmpTuple);
							System.out.println(p);
							patternSet.add(p);
							
						}
					}

				}
			}
		}
				
		out.writeObject(patternSet);
		out.close();
	}
	
	private InstanceList featureEngineering(HashSet<Pattern> patternSet){
		
		InstanceList instances = new InstanceList(new Noop());
		
		for(Pattern p : patternSet){
			FeatureVector fv = fvGenerator.getFeatureVector(p);
			Instance inst = new Instance(fv, null, null, p);
			instances.add(inst);			
		}
		
		return instances;
	}
	
	private InstanceList featureEngineeringOnCategory(String categoryId){
		
		InstanceList seeds = new InstanceList(new Noop());
		Category[] cats = Category.values();
		for(Category cat : cats){
			if(cat.getId() == Integer.parseInt(categoryId)){
				Map<String, String[]> aspects = cat.getAspects(cat.getId());
				Set<String> keys = aspects.keySet();
				for(String key : keys){
					String[] keywords = aspects.get(key);
					FeatureVector fv = fvGenerator.getFeatureVector(keywords);
					Instance inst = new Instance(fv, null, null, key);
					seeds.add(inst);
				}
			}
		}
		
		return seeds;
	}
	
	private void generateFinalSummary( String outputSummaryDir,
			String corpusName, Clustering predicted, InstanceList seeds){
		PrintWriter out = FileOperation.getPrintWriter(new File(outputSummaryDir), corpusName);
		HashSet<InstanceList> set = new HashSet<InstanceList>();
		for(Instance seed : seeds){
			FeatureVector seedFv = (FeatureVector) seed.getData();
			InstanceList[] clusters = predicted.getClusters();
			float Max = Float.MIN_VALUE;
			InstanceList bestCluster = null;
			for(InstanceList cluster : clusters){
				SparseVector meanVec = KMeans.mean(cluster);
				float dist = 0;
				for (int i = 0; i < meanVec.getIndices().length; i++) {
						dist += seedFv.getValues()[i] * meanVec.getValues()[i];
				}
				if(dist >= Max && !set.contains(cluster)){
					Max = dist;
					bestCluster = cluster;
				
				}
			}
			set.add(bestCluster);
			for(int i=0; i < bestCluster.size(); i++){
				Instance inst = bestCluster.get(i);
				Pattern p = (Pattern) inst.getSource();
				CoreMap annotation = p.getCoreMap();
				SemanticGraph graph = annotation.get(BasicDependenciesAnnotation.class);
				String summarySent = realization(p, graph);
				if(summarySent == null)
					continue;
//				String summarySent = p.getCoreMap().toString();
//				out.println(p.toString());
				out.println(summarySent);
				if(i > 1)break;
			}
			out.println();
		}
		
		out.close();
	}

	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, String categoryId) throws Exception {

		InstanceList docs = new InstanceList(pipeLine);
		
/*		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName);
		docs.addThruPipe(fIter); 
		ObjectOutputStream out = new ObjectOutputStream(new 
				FileOutputStream( outputSummaryDir + "/" +corpusName + ".ser")); 
		docs.writeObject(out); 
		out.close();*/
		

/*		System.out.println("Begin generate patterns");
		HeadExtractor headExtractor = new HeadExtractor();
		if(framenetTagger == null)
			framenetTagger = new FramenetTagger();
	    generatePatterns(outputSummaryDir, corpusName, docs, headExtractor);*/
		
		
		
	    System.out.println("Begin summary generation");
	    if(fvGenerator == null)
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
