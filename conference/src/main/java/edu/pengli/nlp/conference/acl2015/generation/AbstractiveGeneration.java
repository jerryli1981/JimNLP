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

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;

import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.extensions.MatlabNumericArray;
import matlabcontrol.extensions.MatlabTypeConverter;
import edu.pengli.nlp.conference.acl2015.pipe.FeatureVectorGenerator;
import edu.pengli.nlp.conference.acl2015.pipe.FramenetTagger;
import edu.pengli.nlp.conference.acl2015.pipe.HeadAnnotation;
import edu.pengli.nlp.conference.acl2015.pipe.WordnetTagger;
import edu.pengli.nlp.conference.acl2015.types.Argument;
import edu.pengli.nlp.conference.acl2015.types.Category;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Predicate;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.algorithms.classify.Clustering;
import edu.pengli.nlp.platform.algorithms.classify.KMeans;
import edu.pengli.nlp.platform.algorithms.classify.SemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.classify.Spectral;
import edu.pengli.nlp.platform.algorithms.miscellaneous.Merger;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;
import edu.pengli.nlp.platform.types.NormalizedDotProductMetric;
import edu.pengli.nlp.platform.types.SparseVector;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RankMap;
import edu.pengli.nlp.platform.util.TimeWait;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
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
	static WordnetTagger wordnetTagger;

	public AbstractiveGeneration() {
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);
	}
	
	private String realization(Pattern p, SemanticGraph graph) {

		SPhraseSpec newSent = nlgFactory.createClause();
		IndexedWord arg1iw = p.getArg1().getHead();
		NPPhraseSpec subjectNp = generateNP(graph, arg1iw);
		newSent.setSubject(subjectNp);

		VPPhraseSpec vp = generateVP(graph, p.getRel(), p.getArg2());
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



	private void generatePatterns(String outputSummaryDir, String corpusName,
			InstanceList corpus, HeadAnnotation headAnnotator) throws Exception {

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".ser"));

		corpus.readObject(in);
		in.close();



		HashSet<Pattern> patternSet = new HashSet<Pattern>();

		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" + corpusName + ".patterns.ser"));
		PrintWriter outt = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".tuples");
		PrintWriter outp = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".patterns");

		int docID = 0;
		for (Instance doc : corpus) {
			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();
			for (CoreMap sent : map.keySet()) {		
				ArrayList<Tuple> tuples = map.get(sent);
				for (Tuple t : tuples) {
											
					ArrayList<IndexedWord> toks = new ArrayList<IndexedWord>();
					toks.addAll(t.getArg1());
					toks.addAll(t.getRel());
					toks.addAll(t.getArg2());
					
					//tuple fusion need know docID to compare IndexedWord
					for(IndexedWord iw : toks){
						iw.setDocID(Integer.toString(docID));
					}
					
					if (t.getRel().lemmatext().equals("say"))
						continue;	
					
					outt.println(t.getSentenceRepresentation());					
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

						//stanford NER tagger
						if(!arg1.getHead().ner().equals("O") && 
								!arg2.getHead().ner().equals("O")){
		
							Pattern p = new Pattern(arg1, pre, arg2, t);
							patternSet.add(p);
							
						}else{
							
							wordnetTagger.annotatePerson(arg1, arg2);
							framenetTagger.annotate(arg1, pre, arg2);
							
							if(arg1.getHead().ner().equals("O")
									|| arg2.getHead().ner().equals("O")){	
								wordnetTagger.annotateNoun(arg1, arg2, t);
							}
							
							if(!arg1.getHead().ner().equals("O") && 
									!arg2.getHead().ner().equals("O")){
								Pattern p = new Pattern(arg1, pre, arg2, t);
								patternSet.add(p);
								outp.println(p.toString());
							}
							
						}
							
					}
				}
			}
			
			docID++;
		}

		
		outt.close();
		out.writeObject(patternSet);
		out.close();
		outp.close();
	}
	
	// if not exist similar vertex to merger, then reture null
	private IndexedWord getSimilarVertex(SemanticGraph graph, IndexedWord vertex){
		
		//the word are appear in the same tuple that have the same docId, sentId, and index
		if(graph.containsVertex(vertex))
			return vertex;
		else{
			//the word are appear in different tuple that have the same mention. 
			String pattern = null;
			if(vertex.tag().startsWith("PRP") || vertex.tag().startsWith("DT")){ //his he should be not alignment. his/PRP$
				pattern = "^"+vertex.originalText();
			}else
				pattern = "^"+vertex.lemma();
			
			List<IndexedWord> similarWords = graph.getAllNodesByWordPattern(pattern);
			if(similarWords.isEmpty())
				return null;
			else{
				IndexedWord iw = similarWords.get(0);
				if(iw.originalText().equals("the") || iw.originalText().equals("to") 
						|| iw.originalText().equals("of") || iw.originalText().equals("have"))
					return null;
				if(iw.docID().equals(vertex.docID()) && iw.sentIndex() == vertex.sentIndex())
					return null;
				else{
					return iw;
				}
					
			}
		}
	}
		
	private ArrayList<ArrayList<IndexedWord>> travelAllPaths(SemanticGraph graph, IndexedWord endNode){
		
		ArrayList<ArrayList<IndexedWord>> ret = new ArrayList<ArrayList<IndexedWord>>();
		
		Stack<IndexedWord> stack = new Stack<IndexedWord>();
		Stack<IndexedWord> path = new Stack<IndexedWord>();
		HashSet<IndexedWord> candidatePoints = new HashSet<IndexedWord>();
		
		//insert START
		stack.add(graph.getFirstRoot());
		boolean stackEmpty = false;		
		boolean pathArriveEnd = false;
		
		while (!stack.isEmpty()) {
			
			
			if(!path.isEmpty() && stack.peek().index() == -2){
				pathArriveEnd = true;
			}
			
			
			boolean containsCandidate = false;
			
			//if path arrive end
			if(pathArriveEnd){

				ArrayList<IndexedWord> pa = new ArrayList<IndexedWord>();
				for(IndexedWord iw : path){
					pa.add(iw);
				}
				ret.add(pa);	
				
				//pop end
				stack.pop();
				if(stack.isEmpty()){
					stackEmpty = true;
					break;
				}
									
				//Backtracking path, using top element of stack to decide backtracking point.
				while(!path.isEmpty()){
					
					if(stackEmpty == false){
						boolean isSibing = graph.getSiblings(path.peek()).contains(stack.peek());
						int isParent = graph.isAncestor(stack.peek(), path.peek());
						
						//reach end
						if(stack.peek().index() == -2){
							pathArriveEnd = true;
							break;
						}
						
						//case 1: if stack.peek is the child of path.peek. then insert
						if(isParent==1 && !path.contains(stack.peek())){
							path.push(stack.peek());
							break;
							
						//case 2: if stack.peek is the sibling of path.peek. then walk towards sibling.
						}else if(isSibing && !path.contains(stack.peek())){
							if(path.size() == 1)
								break;
							
							path.pop();
							
							int parent = graph.isAncestor(stack.peek(), path.peek());
							if(parent == 1){
								path.push(stack.peek());
								break;
							}else{
								//find the parent of stack.peek on path
								do{
									path.pop();
									
								}while(!path.empty() && graph.isAncestor(stack.peek(), path.peek()) != 1);
								
								path.push(stack.peek());
								break;
							}
							
						//case 3: stack.peek() equals path.peek()
						}else if(stack.peek().equals(path.peek())){
							
							if(path.size() == 1)
								break;
							
							
							IndexedWord flag = null;
							do{
								flag = path.pop();
								
							}while(!path.empty()&&path.peek().index() != -1);
							
							
							if(path.empty())
								break;
						
							do{
								stack.pop();
								if(stack.isEmpty()){
									stackEmpty = true;
									break;
								}
								
							}while(!path.empty() && graph.isAncestor(stack.peek(), path.peek()) != 1 
									|| flag.equals(stack.peek()));
							
							if(stackEmpty == false){
								if(!candidatePoints.contains(stack.peek())){
									candidatePoints.add(stack.peek());
									path.push(stack.peek());	
								}else{
									containsCandidate = true;
									//choose candidate stack.peek
									do{
										stack.pop();
										if(stack.isEmpty()){
											stackEmpty = true;
											break;
										}
										
									}while(!path.empty() && graph.isAncestor(stack.peek(), path.peek()) != 1 
											|| flag.equals(stack.peek()) || candidatePoints.contains(stack.peek()));
								}
									
								
							}	
								
							break;
							
						}else
							path.pop();
					}else
						break;
				}
																
			}else{

				if(!path.contains(stack.peek()))
					path.push(stack.peek());
				else{

					//choose another way
					stack.pop();
					
					//Backtracking path, using top element of stack to decide backtracking point.
					while(!path.isEmpty()){
						
						if(stackEmpty == false){
							boolean isSibing = graph.getSiblings(path.peek()).contains(stack.peek());
							int isParent = graph.isAncestor(stack.peek(), path.peek());
							
							//reach end
							if(stack.peek().index() == -2){
								pathArriveEnd = true;
								break;
							}
							
							//case 1: if stack.peek is the child of path.peek. then insert
							if(isParent==1 && !path.contains(stack.peek())){
								path.push(stack.peek());
								break;
								
							//case 2: if stack.peek is the sibling of path.peek. then walk towards sibling.
							}else if(isSibing && !path.contains(stack.peek())){
								if(path.size() == 1)
									break;
								
								path.pop();
								int parent = graph.isAncestor(stack.peek(), path.peek());
								if(parent == 1){
									path.push(stack.peek());
									break;
								}else{
									//find the parent of stack.peek on path
									do{
										path.pop();
										
									}while(graph.isAncestor(stack.peek(), path.peek()) != 1);
									
									path.push(stack.peek());
									break;
								}
								
							//case 3: stack.peek() equals path.peek()
							}else if(stack.peek().equals(path.peek())){
								
								if(path.size() == 1)
									break;

								
								// flag is the next token of start, clear path
								IndexedWord flag = null;
								boolean jump = false;
								do{
									if(graph.isAncestor(endNode, path.peek()) == 1){
										jump = true;
										break;
									}
									flag = path.pop();
									
								}while(path.peek().index() != -1);
								
								if(jump == true){
									pathArriveEnd = true;
									break;
								}	
								
								do{
									stack.pop();
									if(stack.isEmpty()){
										stackEmpty = true;
										break;
									}
									
								}while(graph.isAncestor(stack.peek(), path.peek()) != 1 
										|| flag.equals(stack.peek()));
								
								if(stackEmpty == false){
									if(!candidatePoints.contains(stack.peek())){
										candidatePoints.add(stack.peek());
										path.push(stack.peek());	
									}else{
										containsCandidate = true;
										//choose another candidate stack.peek
										do{
											stack.pop();
											if(stack.isEmpty()){
												stackEmpty = true;
												break;
											}
											
										}while(graph.isAncestor(stack.peek(), path.peek()) != 1 
												|| flag.equals(stack.peek()) || candidatePoints.contains(stack.peek()));
									}
									
								}			
								break;
								
							}else if(graph.isAncestor(endNode, path.peek()) == 1){
								pathArriveEnd = true;
								break;
							}else
								path.pop();
						}else
							break;
					}
				}
			}
			
			if(pathArriveEnd == true)
				continue;
					
			if(containsCandidate == true)
				continue;
			
			if(stackEmpty == true)
				break;
					
			Iterable<SemanticGraphEdge> iter = 
					graph.outgoingEdgeIterable(stack.pop());	
			for (SemanticGraphEdge edge : iter) 
				stack.push(edge.getDependent());		
		}	
		
		return ret;
	}
	private ArrayList<String> patternFusion(InstanceList patternCluster){
				
		//Node Alignment
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
		
		
		//construct graph to merge patterns
		for (int i = 0; i < patternCluster.size(); i++) {
			Instance inst = patternCluster.get(i);
			Pattern p = (Pattern) inst.getSource();
			ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
			// replaced to pattern representation
			for(IndexedWord iw : p.getArg1()){
				if(iw.equals(p.getArg1().getHead())){ 
					iw.setOriginalText(p.getArg1().getHead().ner().toUpperCase().replaceAll(" ", "_"));
				}
				wordList.add(iw);
			}			
			wordList.addAll(p.getRel());		
			for(IndexedWord iw : p.getArg2()){			
				if(iw.equals(p.getArg2().getHead())){
					iw.setOriginalText(p.getArg2().getHead().ner().toUpperCase().replaceAll(" ", "_"));
				}
				wordList.add(iw);
			}

			IndexedWord firstVertex = wordList.get(0);
			IndexedWord flag = getSimilarVertex(graph, firstVertex);
			if(flag == null){
				graph.addEdge(startNode, firstVertex, null, 0.0, false);
			}
		
			for(int j=0; j<wordList.size()-1; j++){	
				IndexedWord source = wordList.get(j);
				IndexedWord flagSource = getSimilarVertex(graph, source);
				IndexedWord dest = wordList.get(j+1);
				IndexedWord flagdest = getSimilarVertex(graph, dest);		
				if(flagSource == null){			
					
					graph.addEdge(source, dest, null, 0.0, false);		
					
				}else if(flagSource != null && flagdest == null){
					
					graph.addEdge(flagSource, dest, null, 0.0, false);	
					
				}else if(flagSource != null && flagdest != null){
					SemanticGraphEdge edge = graph.getEdge(flagSource, flagdest);
					if(edge == null){
						graph.addEdge(flagSource, flagdest, null, 0.0, false);	
					}
				}
			}
			
			IndexedWord lastWord = wordList.get(wordList.size()-1);
			IndexedWord flagLastWord = getSimilarVertex(graph, lastWord);
			IndexedWord flagEndRoot = getSimilarVertex(graph, endNode);
			
			if(flagLastWord != null && flagEndRoot == null){
				
				graph.addEdge(flagLastWord, endNode, null, 0.0, false);	
				
			}else if(flagLastWord != null && flagEndRoot != null){
				SemanticGraphEdge edge = graph.getEdge(flagLastWord, flagEndRoot);
				if(edge == null){
					graph.addEdge(flagLastWord, flagEndRoot, null, 0.0, false);	
				}
			}		

		}	
		
		System.out.println("Begin travel the graph to generate new patterns");
		ArrayList<ArrayList<IndexedWord>> paths = travelAllPaths(graph, endNode);
		
		ArrayList<ArrayList<IndexedWord>> filteredPaths = new 
				ArrayList<ArrayList<IndexedWord>>();
		HashSet<String> set = new HashSet<String>();
		for(int i=0; i<paths.size(); i++){
			ArrayList<IndexedWord> path = paths.get(i);
			StringBuilder sb = new StringBuilder();
			for(IndexedWord iw : path){
				//keep consistent with dictionary
				sb.append(iw.originalText().replaceAll(" ", "_")+" ");
			}
			if(!set.contains(sb.toString().trim())){
				filteredPaths.add(path);
				set.add(sb.toString().trim());
			}
		}
				
		ArrayList<String> merged = new ArrayList<String>();
		for(ArrayList<IndexedWord> path : filteredPaths){
			StringBuilder sb = new StringBuilder();
			for(IndexedWord iw : path){
				//keep consistent with dictionary
				sb.append(iw.originalText().replaceAll(" ", "_")+" ");
			}
			
			//prevent impossible lookup in dictionary
			if(sb.toString().trim().equals("") || sb.toString().trim().equals(" "))
				continue;
			merged.add(sb.toString().trim());
		}
		
		return Merger.process(merged);
	}
	
	private ArrayList<String> tupleFusion(InstanceList tupleCluster){
		
		//Node Alignment
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
		
		
		//construct graph to merge tuples
		for (int i = 0; i < tupleCluster.size(); i++) {
			Instance inst = tupleCluster.get(i);
			Pattern p = (Pattern) inst.getSource();
			Tuple t = p.getTuple();
			ArrayList<IndexedWord> wordList = new ArrayList<IndexedWord>();
			wordList.addAll(t.getArg1());
			wordList.addAll(t.getRel());
			wordList.addAll(t.getArg2());
			IndexedWord firstVertex = wordList.get(0);
			IndexedWord flag = getSimilarVertex(graph, firstVertex);
			if(flag == null){
				graph.addEdge(startNode, firstVertex, null, 0.0, false);
			}
		
			for(int j=0; j<wordList.size()-1; j++){	
				IndexedWord source = wordList.get(j);
				IndexedWord flagSource = getSimilarVertex(graph, source);
				IndexedWord dest = wordList.get(j+1);
				IndexedWord flagdest = getSimilarVertex(graph, dest);		
				if(flagSource == null){			
					
					graph.addEdge(source, dest, null, 0.0, false);		
					
				}else if(flagSource != null && flagdest == null){
					
					graph.addEdge(flagSource, dest, null, 0.0, false);	
					
				}else if(flagSource != null && flagdest != null){
					SemanticGraphEdge edge = graph.getEdge(flagSource, flagdest);
					if(edge == null){
						graph.addEdge(flagSource, flagdest, null, 0.0, false);	
					}
				}
			}
			
			IndexedWord lastWord = wordList.get(wordList.size()-1);
			IndexedWord flagLastWord = getSimilarVertex(graph, lastWord);
			IndexedWord flagEndRoot = getSimilarVertex(graph, endNode);
			
			if(flagLastWord != null && flagEndRoot == null){
				
				graph.addEdge(flagLastWord, endNode, null, 0.0, false);	
				
			}else if(flagLastWord != null && flagEndRoot != null){
				SemanticGraphEdge edge = graph.getEdge(flagLastWord, flagEndRoot);
				if(edge == null){
					graph.addEdge(flagLastWord, flagEndRoot, null, 0.0, false);	
				}
			}		
		}	
		
		System.out.println("Begin travel the graph to generate new tuples");
		ArrayList<ArrayList<IndexedWord>> paths = travelAllPaths(graph, endNode);
		
		ArrayList<ArrayList<IndexedWord>> filteredPaths = new 
				ArrayList<ArrayList<IndexedWord>>();
		HashSet<String> set = new HashSet<String>();
		for(int i=0; i<paths.size(); i++){
			ArrayList<IndexedWord> path = paths.get(i);
			StringBuilder sb = new StringBuilder();
			for(IndexedWord iw : path){
				//keep consistent with dictionary
				sb.append(iw.originalText().replaceAll(" ", "_")+" ");
			}
			if(!set.contains(sb.toString().trim())){
				filteredPaths.add(path);
				set.add(sb.toString().trim());
			}
		}
				
		ArrayList<String> merged = new ArrayList<String>();
		for(ArrayList<IndexedWord> path : filteredPaths){
			StringBuilder sb = new StringBuilder();
			for(IndexedWord iw : path){
				//keep consistent with dictionary
				sb.append(iw.originalText().replaceAll(" ", "_")+" ");
			}
			//prevent impossible lookup in dictionary
			if(sb.toString().trim().equals("") || sb.toString().trim().equals(" "))
				continue;
			merged.add(sb.toString().trim());
		}
		
		return Merger.process(merged);
	}
	
	private HashMap<Instance, Double> getNbestMap(String outputSummaryDir,
			String corpusName, ArrayList<String> candidates, ArrayList<FeatureVector> vectors)
			throws NumberFormatException, IOException {

		PrintWriter nbest = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".nbest");
		
		for(String s : candidates){
			nbest.println(s);
		}
		nbest.close();

		String[] cmd = { "/home/peng/Develop/Workspace/Mavericks/platform/src"
				+ "/main/java/edu/pengli/nlp/platform/algorithms/neuralnetwork/RNNLM/rnnlm",
				"-rnnlm", outputSummaryDir + "/" + corpusName + ".rnnlm.model",
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
		HashMap<Instance, Double> nbestmap = new HashMap<Instance, Double>();
		int i = 0;
		while ((input_nbest = in_nbest.readLine()) != null
				&& (input_score = in_score.readLine()) != null) {
			Instance inst = new Instance(vectors.get(i++), null, null, input_nbest);
			if (input_score.equals("-inf")) {
				nbestmap.put(inst, -100.0);
			} else{
				nbestmap.put(inst, Double.valueOf(input_score));
			}
		}
		in_score.close();
		in_nbest.close();
		return nbestmap;

	}
	
	private String realization(String outputSummaryDir,
			String corpusName, InstanceList patternCluster, MatlabProxy proxy)
					throws FileNotFoundException, 
					IOException, MatlabInvocationException, ClassNotFoundException{
		
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + "ALL" + ".dict.ser"));
		Alphabet dictionary = (Alphabet)in.readObject();
		int maxPatternSize = in.readInt();
		in.close();

		ArrayList<String> tupleCandidates = tupleFusion(patternCluster);
		ArrayList<String> patternCandidates = patternFusion(patternCluster);
		if(tupleCandidates.size() == 0 || patternCandidates.size() == 0){
			System.out.println(" tuple or pattern set is empty");
			System.exit(0);
		}	
		ArrayList<String> candidates  =new ArrayList<String>();
		candidates.addAll(patternCandidates);
		candidates.addAll(tupleCandidates);
		
		ArrayList<String[]> instances  = new ArrayList<String[]>();
		for(String str : candidates){
			String[] inst = new String[2];
			inst[0] = str;
			inst[1] = "1";
			instances.add(inst);
		}

		String matInputFile = outputSummaryDir + "/" + corpusName + "_In_AllPosi.mat";
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
		
		String matInputFile_Final = outputSummaryDir + "/" + corpusName + "_In_Final.mat";
		new MatFileWriter(matInputFile_Final, list);
		
		String modelOutputFile = outputSummaryDir + "/" + "ALL" + "_Model.mat";
		String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out_final.mat";
				
		ArrayList<FeatureVector> patternTupleVectors = FeatureVectorGenerator.getVectors(modelOutputFile, matInputFile_Final,
				matOutputFile, proxy);
		
		ArrayList<FeatureVector> patternVectors = new ArrayList<FeatureVector>();
		ArrayList<FeatureVector> tupleVectors = new ArrayList<FeatureVector>();
		int k=0;
		for(int i=0; i<patternCandidates.size(); i++){
			patternVectors.add(patternTupleVectors.get(k++));
		}
		for(int j=0; j<tupleCandidates.size(); j++){
			tupleVectors.add(patternTupleVectors.get(k++));
		}
		
		HashMap<Instance, Double> nbestMap = getNbestMap(outputSummaryDir, corpusName, 
				patternCandidates, patternVectors);
		LinkedHashMap rankedmap = RankMap.sortHashMapByValues(nbestMap, true);
		Set<Instance> keys = rankedmap.keySet();
		Iterator iter = keys.iterator();
		FeatureVector bestPatternVector = null;
		if(iter.hasNext()){
			Instance bestPatternInst = (Instance)iter.next();
			bestPatternVector = (FeatureVector)bestPatternInst.getData();
		}
		int idx=0;
		double min = Double.MAX_VALUE;
		
		double[] vec_p = new double[bestPatternVector.getValues().length];
		double len_p = 0;
		for (int a = 0; a < bestPatternVector.getValues().length; a++) {
			len_p += bestPatternVector.getValues()[a] * 
					bestPatternVector.getValues()[a];
		}
		len_p = (double) Math.sqrt(len_p);
		for (int a = 0; a < bestPatternVector.getValues().length; a++) {
			vec_p[a] /= len_p;
		}
		
		for(int i=0; i<tupleVectors.size(); i++){
			FeatureVector tfv = tupleVectors.get(i);
			double[] vec_t = new double[tfv.getValues().length];
			double len = 0;
			for (int a = 0; a < tfv.getValues().length; a++) {
				len += tfv.getValues()[a] * tfv.getValues()[a];
			}
			len = (double) Math.sqrt(len);
			for (int a = 0; a < tfv.getValues().length; a++) {
				vec_t[a] /= len;
			}
			
			double dist = 0.0;
			for (int j = 0; j < tfv.getValues().length; j++) {
				dist += vec_t[j] * vec_p[j];
			}
			if(dist < min){
				min = dist;
				idx = i;
			}	
		}
		
		return tupleCandidates.get(idx);
	}

	private void generateFinalSummary(String outputSummaryDir,
			String corpusName, Clustering predicted, InstanceList seeds, MatlabProxy proxy) 
					throws NumberFormatException, IOException, 
					MatlabInvocationException, ClassNotFoundException {
		
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);
		HashSet<InstanceList> set = new HashSet<InstanceList>();
		
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".dict.ser"));
		Alphabet dictionary = (Alphabet)in.readObject();
		int maxPatternSize = in.readInt();
		in.close();

		int seedIdx = 0;
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
			if(bestCluster == null)
				continue;
			

			System.out.println("Tuple fusion in seed "+seedIdx);
			ArrayList<String> tupleCandidates = tupleFusion(bestCluster);
			System.out.println("Pattern fusion in seed "+seedIdx);
			seedIdx++;
			ArrayList<String> patternCandidates = patternFusion(bestCluster);
			
			ArrayList<String> candidates  =new ArrayList<String>();
			candidates.addAll(patternCandidates);
			candidates.addAll(tupleCandidates);
			
			ArrayList<String[]> instances  = new ArrayList<String[]>();
			for(String str : candidates){
				String[] inst = new String[2];
				inst[0] = str;
				inst[1] = "1";
				instances.add(inst);
			}
			
			String modelOutputFile = outputSummaryDir + "/" + corpusName + "_Model.mat";
			String matOutputFile = outputSummaryDir + "/" + corpusName + "_Out_final.mat";
			
			String matInputFile2 = outputSummaryDir + "/" + corpusName + "_In2.mat";
			MatFileReader red = new MatFileReader(matInputFile2);
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
			
			String matInputFile_Final = outputSummaryDir + "/" + corpusName + "_In_Final.mat";
			new MatFileWriter(matInputFile_Final, list);
					
			ArrayList<FeatureVector> patternTupleVectors = FeatureVectorGenerator.getVectors(modelOutputFile, matInputFile_Final,
					matOutputFile,  proxy);
			
			ArrayList<FeatureVector> patternVectors = new ArrayList<FeatureVector>();
			ArrayList<FeatureVector> tupleVectors = new ArrayList<FeatureVector>();
			int k=0;
			for(int i=0; i<patternCandidates.size(); i++){
				patternVectors.add(patternTupleVectors.get(k++));
			}
			for(int j=0; j<tupleCandidates.size(); j++){
				tupleVectors.add(patternTupleVectors.get(k++));
			}
			
			HashMap<Instance, Double> nbestMap = getNbestMap(outputSummaryDir, corpusName, 
					patternCandidates, patternVectors);
			LinkedHashMap rankedmap = RankMap.sortHashMapByValues(nbestMap, true);
			Set<Instance> keys = rankedmap.keySet();
			Iterator iter = keys.iterator();
			FeatureVector bestPatternVector = null;
			if(iter.hasNext()){
				Instance bestPatternInst = (Instance)iter.next();
				bestPatternVector = (FeatureVector)bestPatternInst.getData();
			}
			int idx=0;
			double min = Double.MAX_VALUE;
			
			double[] vec_p = new double[bestPatternVector.getValues().length];
			double len_p = 0;
			for (int a = 0; a < bestPatternVector.getValues().length; a++) {
				len_p += bestPatternVector.getValues()[a] * 
						bestPatternVector.getValues()[a];
			}
			len_p = (double) Math.sqrt(len_p);
			for (int a = 0; a < bestPatternVector.getValues().length; a++) {
				vec_p[a] /= len_p;
			}
			
			for(int i=0; i<tupleVectors.size(); i++){
				FeatureVector tfv = tupleVectors.get(i);
				double[] vec_t = new double[tfv.getValues().length];
				double len = 0;
				for (int a = 0; a < tfv.getValues().length; a++) {
					len += tfv.getValues()[a] * tfv.getValues()[a];
				}
				len = (double) Math.sqrt(len);
				for (int a = 0; a < tfv.getValues().length; a++) {
					vec_t[a] /= len;
				}
				
				double dist = 0.0;
				for (int j = 0; j < tfv.getValues().length; j++) {
					dist += vec_t[j] * vec_p[j];
				}
				if(dist < min){
					min = dist;
					idx = i;
				}	
			}
			out.println(tupleCandidates.get(idx));
		}
		
		out.close();
	}
	
	private InstanceList[] kmeans(InstanceList instances, int numClusters){
		Metric metric = new NormalizedDotProductMetric();
		KMeans kmeans = new KMeans(new Noop(), numClusters, metric);
		Clustering predicted = kmeans.cluster(instances);
		return predicted.getClusters();
	}
	
	private InstanceList[] spectral(InstanceList instances, int numClusters, MatlabProxy proxy){
		Metric metric = new NormalizedDotProductMetric();
		Spectral spectral = new Spectral(new Noop(), numClusters, metric, proxy);
		Clustering predicted = spectral.cluster(instances); 
		return predicted.getClusters();
	}
	
	private InstanceList[] seedBasedClustering(InstanceList 
			instances, String categoryId, MatlabProxy proxy){
		
		int dimension = 20;
		InstanceList seeds = new InstanceList(new Noop());
		Category[] cats = Category.values();
		for (Category cat : cats) {
			if (cat.getId() == Integer.parseInt(categoryId)) {
				Map<String, String[]> aspects = cat.getAspects(cat.getId());
				Set<String> keys = aspects.keySet();
				for (String key : keys) {
					String[] keywords = aspects.get(key);
					FeatureVector fv = FeatureVectorGenerator.getFeatureVector(keywords, dimension);
					Instance inst = new Instance(fv, null, null, key);
					seeds.add(inst);
				}
			}
		}
		
		Metric metric = new NormalizedDotProductMetric();
		SemiSupervisedClustering semiClustering = new 
				SemiSupervisedClustering(new Noop(), seeds,  metric, proxy);
		Clustering predicted = semiClustering.cluster(instances); 
		return predicted.getClusters();	
	}
	
	private void generateFinalSummary_X(String outputSummaryDir,
			String corpusName, InstanceList instances, String categoryId, MatlabProxy proxy) 
					throws NumberFormatException, IOException, 
					MatlabInvocationException, ClassNotFoundException {
		
		System.out.println("Begin pattern clustering");
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);
			
		int numClusters  = 4;

		//method 1: kmeans unsupervised clustering
		//ROUGE-SU4 is 0.06278 (k = 4)
		//ROUGE-SU4 is 0.0706 (k = 5)
		//ROUGE-SU4 is 0.08875 (k = 6)
		//ROUGE-SU4 is 0.09603 (k = 7)
		InstanceList[] groups_k = kmeans(instances, numClusters);
		for(InstanceList cluster :groups_k){
			out.println(realization(outputSummaryDir,
					corpusName, cluster,  proxy));
		}
		
		//method 2: spectral unsupervised clustering
		//ROUGE-SU4 is 0.03736 (spectral k=4)
		//ROUGE-SU4 is 0.05479 (spectral k=5)
		//ROUGE-SU4 is 0.05715 (spectral k=6)
		//ROUGE-SU4 is 0.08172 (spectral k=7)
/*		InstanceList[] groups_s = spectral(instances, numClusters, proxy);
		for(InstanceList cluster :groups_s){
			out.println(realization(outputSummaryDir,
					corpusName, cluster,  proxy));
		}*/
		
		//method 3: seed based semi-supervised clusterting
 		//ROUGE-SU4 is ? (k=4)
		//ROUGE-SU4 is ? (k=5)
		//ROUGE-SU4 is ? (k=6)
		//ROUGE-SU4 is ? (k=7)
/*		InstanceList[] groups_seed = seedBasedClustering(instances, categoryId,  proxy);
		for(InstanceList cluster :groups_seed){
			out.println(realization(outputSummaryDir,
					corpusName, cluster,  proxy));
		}*/
		
		out.close();
	}
	
	private void trainRNN(String outputSummaryDir,
			String corpusName) throws IOException{
		System.out.println("train RNNLM to scoring generated patterns");
		PrintWriter out_valid = new PrintWriter(new 
				FileOutputStream(new File(outputSummaryDir + "/" + corpusName + ".patterns.valid")));
		
		BufferedReader in_train =new BufferedReader(new FileReader(
				new File(outputSummaryDir + "/" + corpusName + ".patterns")));
		ArrayList<String> trainsents = new ArrayList<String>();
		String input = null;
		while((input=in_train.readLine()) != null){
			trainsents.add(input);
		}
		in_train.close();
		Random rand = new Random();
		int size = trainsents.size();
		int newSize = size;
		for(int i=0; i<size*0.2; i++){
			int ran = rand.nextInt(newSize);
			out_valid.println(trainsents.get(ran));
			trainsents.remove(ran);
			newSize--;
		}
		out_valid.close();
		String[] cmd = {"/home/peng/Develop/Workspace/Mavericks/platform/src"
				+ "/main/java/edu/pengli/nlp/platform/algorithms/neuralnetwork/RNNLM/rnnlm", 
				"-train", outputSummaryDir + "/" + corpusName + ".patterns", "-valid", 
				outputSummaryDir + "/" + corpusName + ".patterns.valid", "-rnnlm", 
				outputSummaryDir + "/" + corpusName + ".rnnlm.model", "-hidden", "40", "-rand-seed", "1",
				"-debug", "2", "-bptt", "3", "-class", "200"};
		
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
	
	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, String categoryId, 
			MatlabProxy proxy)
			throws Exception {

		InstanceList docs = new InstanceList(pipeLine);

		
/*    	OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName); docs.addThruPipe(fIter);
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" +corpusName + ".ser")); docs.writeObject(out);	
		out.close();*/
		
/*		System.out.println("Begin generate patterns"); 
		HeadAnnotation headAnnotator = new HeadAnnotation(); 
		if(framenetTagger == null)
			framenetTagger = new FramenetTagger();
		if(wordnetTagger == null)
			wordnetTagger = new WordnetTagger();
		generatePatterns(outputSummaryDir, corpusName, docs, headAnnotator );*/


		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".patterns.ser"));
		HashSet<Pattern> patternSet = (HashSet<Pattern>) in.readObject();
		in.close();
		
		System.out.println("Begin pattern representation learning");
		InstanceList patternList = new InstanceList(new Noop());
		for (Pattern p : patternSet) {
			Instance inst = new Instance(p, null, null, p);
			patternList.add(inst);
		}
		InstanceList instances = new InstanceList(pipeLine);
		FeatureVectorGenerator fvGenerator = 
				(FeatureVectorGenerator) pipeLine.getPipe(0);
		//ROUGE-SU4 is 0.03367 by general pattern
		//ROUGE-SU4 is 0.05414 by specific pattern, release rel, keep arg the same as general pattern
		//ROUGE-SU4 is 0.09934 by specific pattern, release rel, release arg other parts replace head with type
		//ROUGE-SU4 is 0.05651 by specific pattern, keep rel head, release arg other parts replace head with type
		//ROUGE-SU4 is 0.0643  by tuple
//		fvGenerator.batchGenerateVectorsByGeneralPatterns(outputSummaryDir, corpusName, patternList, proxy);
		
		//if choose this, then choose generateFinalSummary
//		fvGenerator.batchGenerateVectorsBySpecificPatterns(outputSummaryDir, corpusName, patternList, proxy);
		
		//if choose this, then choose generateFinalSummary_X
        fvGenerator.batchGetVectors(outputSummaryDir, corpusName, patternList, proxy);
//		fvGenerator.batchGenerateVectorsByTuples(outputSummaryDir, corpusName, patternList, proxy);
		instances.addThruPipe(patternList.iterator());
		
		System.out.println("Begin generate final summary");
//		generateFinalSummary(outputSummaryDir, corpusName, predicted, seeds, proxy);
		generateFinalSummary_X(outputSummaryDir, corpusName, instances, categoryId, proxy);

	}
}
