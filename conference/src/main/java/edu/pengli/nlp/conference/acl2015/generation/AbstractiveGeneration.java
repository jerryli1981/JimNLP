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

import com.jmatio.io.MatFileReader;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLCell;
import com.jmatio.types.MLDouble;

import lpsolve.LpSolve;
import matlabcontrol.MatlabConnectionException;
import matlabcontrol.MatlabInvocationException;
import matlabcontrol.MatlabProxy;
import matlabcontrol.MatlabProxyFactory;
import matlabcontrol.MatlabProxyFactoryOptions;
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
import edu.pengli.nlp.platform.algorithms.classify.HarmonicSemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.classify.KMeans_Java;
import edu.pengli.nlp.platform.algorithms.classify.KMeans_Matlab;
import edu.pengli.nlp.platform.algorithms.classify.LabelPropagationSemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.classify.LocalglobalConsistencySemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.classify.SemiSupervisedClustering;
import edu.pengli.nlp.platform.algorithms.classify.Spectral_Matlab;
import edu.pengli.nlp.platform.algorithms.classify.Spectral_Java;
import edu.pengli.nlp.platform.algorithms.miscellaneous.LongestCommonSubstring;
import edu.pengli.nlp.platform.algorithms.miscellaneous.Merger;
import edu.pengli.nlp.platform.algorithms.ranking.LexRank;
import edu.pengli.nlp.platform.pipe.CharSequence2TokenSequence;
import edu.pengli.nlp.platform.pipe.FeatureSequence2FeatureVector;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.TokenSequence2FeatureSequence;
import edu.pengli.nlp.platform.pipe.TokenSequenceLowercase;
import edu.pengli.nlp.platform.pipe.TokenSequenceRemoveStopwords;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.FeatureVector;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.types.Metric;
import edu.pengli.nlp.platform.types.NormalizedDotProductMetric;
import edu.pengli.nlp.platform.types.SparseVector;
import edu.pengli.nlp.platform.util.CallableTask;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.Maths;
import edu.pengli.nlp.platform.util.RankMap;
import edu.pengli.nlp.platform.util.TimeWait;
import edu.pengli.nlp.platform.util.matrix.Matrix;
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
					/*
					 * if (obj != null) { PPPhraseSpec ppp =
					 * generatePrepP(graph, prep, obj); if
					 * (np.getPostModifiers().size() != 0) {
					 * np.addPostModifier(ppp); } else np.setPostModifier(ppp);
					 * }
					 */

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
						|| gr.toString().equals("poss")
						|| gr.toString().equals("neg")) {
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

		// here is the whole pred mention as vp head, but why? keep readability?
		vp.setHead(pred.originaltext());

		/*
		 * // set aux of the headVerb Iterable<SemanticGraphEdge> children =
		 * graph .outgoingEdgeIterable(headVp); for (SemanticGraphEdge edge :
		 * children) { GrammaticalRelation gr = edge.getRelation(); if
		 * (gr.toString().equals("aux")) {
		 * vp.setPreModifier(edge.getDependent().originalText()); break; } }
		 */

		// using the shortest dependency path between headVp and head Argument
		// to decide generation type

		// already test whether headVp is null when pattern generation. so
		// here do not need to test
		IndexedWord headVp = pred.getHead();
		IndexedWord argHead = arg2.getHead();
		ArrayList<IndexedWord> toks = new ArrayList<IndexedWord>();
		for (int i = headVp.index(); i < arg2.get(arg2.size() - 1).index(); i++) {
			IndexedWord word = graph.getNodeByIndexSafe(i);
			if (word != null)
				toks.add(word);
		}

		List<SemanticGraphEdge> path = graph.getShortestDirectedPathEdges(
				headVp, argHead);
		/*
		 * List<SemanticGraphEdge> path = new ArrayList<SemanticGraphEdge>();
		 * for (int i = 0; i < toks.size(); i++) { for (int j = 0; j <
		 * toks.size(); j++) { if (i == j) continue; else { IndexedWord ai =
		 * toks.get(i); IndexedWord aj = toks.get(j); if (ai == null || aj ==
		 * null) continue; List<SemanticGraphEdge> edge = graph.getAllEdges(ai,
		 * aj); path.addAll(edge); } } }
		 */

		if (path == null) {
			vp.setObject(arg2.toString());
			return vp;
		}

		IndexedWord prep = null;
		IndexedWord dobj = null;
		IndexedWord pobj = null;
		for (SemanticGraphEdge edge : path) {
			GrammaticalRelation dgr = edge.getRelation();
			if (dgr.toString().equals("pobj")) {
				prep = edge.getGovernor();
				pobj = edge.getDependent();
				PPPhraseSpec ppp = generatePrepP(graph, prep, pobj);
				vp.addPostModifier(ppp);

			} else if (dgr.toString().equals("dobj")) {
				dobj = edge.getDependent();

				NPPhraseSpec dirObjNp = generateNP(graph, dobj);
				vp.setObject(dirObjNp);
			} else if (dgr.toString().endsWith("comp")) {
				NPPhraseSpec np = generateNP(graph, edge.getDependent());
				vp.addComplement(np);
			} else if (dgr.toString().endsWith("mod")
					&& edge.getGovernor().equals(headVp)) {
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

					// tuple fusion need know docID to compare IndexedWord
					for (IndexedWord iw : toks) {
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

					// for later sentence realization to get head verb
					edu.pengli.nlp.conference.acl2015.types.Predicate pre = headAnnotator
							.annotatePredicateHead(t.getRel(), sent);
					t.setRel(pre);

					// for complicated arguments, just ignore, so arg may be
					// null
					if (arg1 == null || arg2 == null)
						continue;

					// tuple with no head should not go into clustering ?
					if (pre.getHead() == null)
						continue;

					if (arg1.getHead() != null && arg2.getHead() != null) {

						// stanford NER tagger
						if (!arg1.getHead().ner().equals("O")
								&& !arg2.getHead().ner().equals("O")) {

							Pattern p = new Pattern(arg1, pre, arg2, t);
							patternSet.add(p);

						} else {

							wordnetTagger.annotatePerson(arg1, arg2);
							framenetTagger.annotate(arg1, pre, arg2);

							if (arg1.getHead().ner().equals("O")
									|| arg2.getHead().ner().equals("O")) {
								wordnetTagger.annotateNoun(arg1, arg2, t);
							}

							if (!arg1.getHead().ner().equals("O")
									&& !arg2.getHead().ner().equals("O")) {
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

	private ArrayList<String> patternFusion(InstanceList patternCluster) throws NoSuchMethodException, SecurityException {

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

		// construct graph to merge patterns
		ArrayList<IndexedWord> lastWords = new ArrayList<IndexedWord>();
		for (int i = 0; i < patternCluster.size(); i++) {
			Instance inst = patternCluster.get(i);
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
				lastWords.add(flagLastWord);

			} else if (flagLastWord != null && flagEndRoot != null) {
				SemanticGraphEdge edge = graph.getEdge(flagLastWord,
						flagEndRoot);
				if (edge == null) {
					graph.addEdge(flagLastWord, flagEndRoot, null, 0.0, false);
					lastWords.add(flagLastWord);
				}
			}
			
			

		}
		
		ArrayList<ArrayList<IndexedWord>> paths = new ArrayList<ArrayList<IndexedWord>>();
		for(IndexedWord last : lastWords){
			List<IndexedWord> path = graph.getPathToRoot(last);
			path.add(0, last);
			ArrayList<IndexedWord> reversePath = new ArrayList<IndexedWord>();
			for(int i = path.size()-2; i>=0; i--){
				reversePath.add(path.get(i));
			}
			paths.add(reversePath);
		}

//		System.out.println("Begin travel the graph to generate new patterns");
//		ArrayList<ArrayList<IndexedWord>> paths = travelAllPaths(graph, endNode);
/*		Method method = getClass().getDeclaredMethod("travelAllPaths", 
				new Class[]{SemanticGraph.class, IndexedWord.class});
		List<Object> args = new ArrayList<Object>();
		args.add(graph);
		args.add(endNode);
		Callable call = new CallableTask(this, method, args);
		FutureTask task = new FutureTask(call);
		
		Thread thread = new Thread(task);
		thread.setDaemon(true);
		thread.start();
		ArrayList<ArrayList<IndexedWord>> paths = null;
		try{
			
			paths = (ArrayList<ArrayList<IndexedWord>>) task.get(20, TimeUnit.SECONDS);
			
		}catch(Exception e){
			System.out.println("pattern fusion can't be fininshed in time");
			return null;
		}*/

		ArrayList<ArrayList<IndexedWord>> filteredPaths = new ArrayList<ArrayList<IndexedWord>>();
		HashSet<String> set = new HashSet<String>();
		for (int i = 0; i < paths.size(); i++) {
			ArrayList<IndexedWord> path = paths.get(i);
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
		for (ArrayList<IndexedWord> path : filteredPaths) {
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

	private ArrayList<String> tupleFusion(InstanceList tupleCluster) throws NoSuchMethodException, SecurityException {

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

		// construct graph to merge tuples
//		ArrayList<IndexedWord> lastWords = new ArrayList<IndexedWord>();
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
//				lastWords.add(flagLastWord);

			} else if (flagLastWord != null && flagEndRoot != null) {
				SemanticGraphEdge edge = graph.getEdge(flagLastWord,
						flagEndRoot);
				if (edge == null) {
					graph.addEdge(flagLastWord, flagEndRoot, null, 0.0, false);
//					lastWords.add(flagLastWord);
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
		Method method = getClass().getDeclaredMethod("travelAllPaths", 
				new Class[]{SemanticGraph.class, IndexedWord.class});
		List<Object> args = new ArrayList<Object>();
		args.add(graph);
		args.add(endNode);
		Callable call = new CallableTask(this, method, args);
		FutureTask task = new FutureTask(call);
		
		Thread thread = new Thread(task);
		thread.setDaemon(true);
		thread.start();
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

	private HashMap<Instance, Double> getNbestMap(String outputSummaryDir,
			String corpusName, ArrayList<String> candidates) throws NumberFormatException,
			IOException {

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
		return nbestmap;

	}

	private ArrayList<String> realization(String outputSummaryDir,
			String corpusName, InstanceList patternCluster, MatlabProxy proxy,
			FeatureVectorGenerator fvGenerator) throws FileNotFoundException,
			IOException, MatlabInvocationException, ClassNotFoundException, NoSuchMethodException, SecurityException {
		
		//1. find the best pattern
		ArrayList<String> tupleCandidates = tupleFusion(patternCluster);
		if(tupleCandidates == null)
			return null;
		ArrayList<String> patternCandidates = patternFusion(patternCluster);	
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
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".smallWordMap"));
		HashMap<String, float[]> wordMap = (HashMap<String, float[]>)in.readObject();
		in.close();
//		HashMap<String, float[]> wordMap = fvGenerator.getWordMap();
		Metric metric = new NormalizedDotProductMetric();
		for(String tuple : tupleCandidates){
			double coverageScore = 0.0;
			String[] tokTup = tuple.split(" ");
			String[] tokPat = bestPattern.split(" ");
			for(String tokT : tokTup)
				for(String tokP : tokPat){
					tokP = fvGenerator.cleaning(tokP.toLowerCase());
					float[] wordVector_P = wordMap.get(tokP);
					
					tokT = fvGenerator.cleaning(tokT.toLowerCase());
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
		ArrayList<String> ret = new ArrayList<String>();
		Set<String> ts = sortedScores.keySet();
		Iterator its = ts.iterator();
		int retNumber = 0;
		while(its.hasNext() && retNumber < 1){
			ret.add((String)its.next());
			retNumber++;
		}
		return ret;
	}
	
	private Object[] seedBasedClustering(String outputSummaryDir, String corpusName,
			InstanceList instances, String categoryId, MatlabProxy proxy,
			FeatureVectorGenerator fvGenerator, int sigma) throws ClassNotFoundException,
			IOException, MatlabInvocationException{
		
//		HashMap<String, float[]> wordMap = fvGenerator.getWordMap();
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".smallWordMap"));
		HashMap<String, float[]> wordMap = (HashMap<String, float[]>)in.readObject();

		in.close();
		
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

		// below code for plot for debug InstanceList allInsts = new
		/*
		 * InstanceList allInsts = new InstanceList(null); for(Instance seed :
		 * seeds) allInsts.add(seed); for(Instance inst : instances)
		 * allInsts.add(inst);
		 * 
		 * int dim = 900; double[][] points = new
		 * double[instances.size()+seeds.size()][dim];
		 * 
		 * double[] arr = new double[(instances.size()+seeds.size())*dim];
		 * 
		 * 
		 * for (int i = 0; i < allInsts.size(); i++) { FeatureVector fv_i =
		 * (FeatureVector) allInsts.get(i).getData(); for(int l=0; l<dim; l++)
		 * points[i][l] = fv_i.getValues()[l]; }
		 * 
		 * int c = 0; for(int l=0; l<dim; l++){ for (int i = 0; i
		 * <allInsts.size(); i++) { arr[c++] = points[i][l]; } } ArrayList list
		 * = new ArrayList(); list.add(new MLDouble("D", arr, allInsts.size()));
		 * String matInputFile_AllPosi = "/home/peng/Downloads/visual/D.mat";
		 * try { new MatFileWriter(matInputFile_AllPosi, list); } catch
		 * (IOException e) { // TODO Auto-generated catch block
		 * e.printStackTrace(); }
		 */

			
/*		SemiSupervisedClustering semiClustering = new 
				HarmonicSemiSupervisedClustering(
				new Noop(), seeds, metric, proxy, sigma);*/
		
/*		SemiSupervisedClustering semiClustering = new 
				LocalglobalConsistencySemiSupervisedClustering(
				new Noop(), seeds, metric, proxy, sigma);*/
		
		SemiSupervisedClustering semiClustering = new 
				LabelPropagationSemiSupervisedClustering(
				new Noop(), seeds, metric, proxy, sigma);
		
		Clustering predicted = semiClustering.cluster(instances);
		Object[] ret = new Object[2];
		ret[0] = predicted;
		ret[1] = seeds;
		return ret;
	}

	private void generateFinalSummary(String outputSummaryDir,
			String corpusName, InstanceList instances, String categoryId,
			MatlabProxy proxy, FeatureVectorGenerator fvGenerator, int sigma)
			throws NumberFormatException, IOException,
			MatlabInvocationException, ClassNotFoundException, NoSuchMethodException, SecurityException{

//		System.out.println("Begin to train RNN to score patterns");
//		trainRNN(outputSummaryDir, corpusName);

		
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);

//		int numClusters = 25;
		int length = 0;
		boolean flag = false;
		// method 1: kmeans unsupervised clustering
		// ROUGE-SU4 is 0.09306 (k = 25), ROUGE-2 is 0.05433, ROUGE-1 is 0.30567
		// ROUGE-SU4 is 0.093573 (k = 30), ROUGE-2 is 0.055126, ROUGE-1 is 0.30468
		// ROUGE-SU4 is 0.09244 (k = 35), ROUGE-2 is ?, ROUGE-1 is ?
/*		InstanceList[] groups_k = kmeans(instances, numClusters, proxy);
		HashMap<InstanceList, Integer> tmp = new HashMap<InstanceList, Integer>();
		for (InstanceList il : groups_k)
			tmp.put(il, il.size());
		HashMap rankedMap = RankMap.sortHashMapByValues(tmp, false);
		Set keys = rankedMap.keySet();
		Iterator iter = keys.iterator();

		while (iter.hasNext()) {
			InstanceList il = (InstanceList) iter.next();
			ArrayList<String> lines = realization(outputSummaryDir, corpusName,
					il, proxy, fvGenerator);
			if (lines == null)
				continue;
			for (String line : lines) {
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
			}

			if (flag == true)
				break;
		}*/

		// method 2: spectral unsupervised clustering
		// ROUGE-SU4 is 0.066083 (k = 5), ROUGE-2 is 0.03363, ROUGE-1 is 0.216196
		// ROUGE-SU4 is 0.089286 (k = 10), ROUGE-2 is 0.05308, ROUGE-1 is 0.299493
		// ROUGE-SU4 is 0.0925066 (k = 15), ROUGE-2 is 0.053203, ROUGE-1 is 0.30444 
		// ROUGE-SU4 is 0.0901866 (k = 20), ROUGE-2 is 0.053219(0.05314), ROUGE-1 is 0.30414
		// ROUGE-SU4 is 0.091803 (0.09591) (k = 25), ROUGE-2 is 0.05639(0.052493), ROUGE-1 is 0.30768
		// ROUGE-SU4 is 0.09262 (0.094746) (k = 30), ROUGE-2 is 0.053796(0.052106), ROUGE-1 is 0.30784
		// ROUGE-SU4 is 0.092339 (k = 35), ROUGE-2 is 0.05696(0.054473), ROUGE-1 is 0.30273
		// (k=40) ROUGE-2 is 0.052753

/*		InstanceList[] groups_s = spectral(instances, numClusters, proxy);
		HashMap<InstanceList, Integer> tmp = new HashMap<InstanceList, Integer>();
		for (InstanceList il : groups_s)
			tmp.put(il, il.size());
		HashMap rankedMap = RankMap.sortHashMapByValues(tmp, false);
		Set keys = rankedMap.keySet();
		Iterator iter = keys.iterator();
		while (iter.hasNext()) {
			InstanceList il = (InstanceList) iter.next();
			ArrayList<String> lines = realization(outputSummaryDir, corpusName,
					il, proxy, fvGenerator);
			if (lines == null)
				continue;
			for (String line : lines) {
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
			}

			if (flag == true)
				break;
		}*/

		// method 3: seed based semi-supervised clusterting
		Object[] objs = seedBasedClustering(outputSummaryDir, corpusName,
				instances, categoryId, proxy, fvGenerator, sigma);
		Clustering predicted = (Clustering) objs[0];
		InstanceList seeds = (InstanceList) objs[1];
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
						ArrayList<String> lines = realization(outputSummaryDir, corpusName,
								il, proxy, fvGenerator);
						if (lines == null)
							continue;
						for (String line : lines) {
							String[] toks = line.split(" ");
							length += toks.length;
							if (length <= 100){
								out.println(line);
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
						}

						if (flag == true)
							break;
					}
					
				}	
				
			}//end if find aspects
			
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
	


	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, String categoryId,
			MatlabProxy proxy, int sigma) throws Exception {

//		InstanceList docs = new InstanceList(pipeLine);

		/*
		 * OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
		 * inputCorpusDir + "/" + corpusName); docs.addThruPipe(fIter);
		 * ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
		 * outputSummaryDir + "/" +corpusName + ".ser")); docs.writeObject(out);
		 * out.close();
		 */

		/*
		 * System.out.println("Begin generate patterns"); HeadAnnotation
		 * headAnnotator = new HeadAnnotation(); if(framenetTagger == null)
		 * framenetTagger = new FramenetTagger(); if(wordnetTagger == null)
		 * wordnetTagger = new WordnetTagger();
		 * generatePatterns(outputSummaryDir, corpusName, docs, headAnnotator );
		 */

/*		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".patterns.ser"));
		HashSet<Pattern> patternSet = (HashSet<Pattern>) in.readObject();
		in.close();

		InstanceList patternList = new InstanceList(new Noop());
		HashSet<String> set = new HashSet<String>();
		for (Pattern p : patternSet) {
			Instance inst = new Instance(p, null, null, p);
			if(!set.contains(p.toString())){
				patternList.add(inst);
				set.add(p.toString());
			}
			
		}*/
		InstanceList instances = new InstanceList(pipeLine);
		FeatureVectorGenerator fvGenerator = (FeatureVectorGenerator) pipeLine
				.getPipe(0);
		
/*		fvGenerator.generateSmallWordVector
		       (outputSummaryDir, corpusName, patternList, categoryId);*/

//		System.out.println("Begin generate feature vectors for patterns");
		// fvGenerator.setFvsViaTrainedDCNN(outputSummaryDir, corpusName,
		// patternList, proxy);
/*		fvGenerator.setFvsViaPreTrainedWord2VecModel(outputSummaryDir,
				corpusName, patternList);

		instances.addThruPipe(patternList.iterator());
		
		ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(
				outputSummaryDir + "/" + corpusName + ".featuredInsts"));
		instances.writeObject(out);
		out.close();*/
		
		ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".featuredInsts"));
		instances.readObject(inInst);
		inInst.close();
		
//		System.out.println("Begin generate final summary");
		generateFinalSummary(outputSummaryDir, corpusName, instances,
				categoryId, proxy, fvGenerator, sigma);

	}
	
	public void run_SimpleNLG(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, String categoryId,
			MatlabProxy proxy, int sigma) throws Exception {

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".patterns.ser"));
		HashSet<Pattern> patternSet = (HashSet<Pattern>) in.readObject();
		in.close();

		InstanceList patternList = new InstanceList(new Noop());
		for (Pattern p : patternSet) {
			Instance inst = new Instance(p, null, null, p);
			patternList.add(inst);
		}
		InstanceList instances = new InstanceList(pipeLine);
		FeatureVectorGenerator fvGenerator = (FeatureVectorGenerator) pipeLine
				.getPipe(0);

//		System.out.println("Begin generate feature vectors for patterns");
		// fvGenerator.setFvsViaTrainedDCNN(outputSummaryDir, corpusName,
		// patternList, proxy);
/*		fvGenerator.setFvsViaPreTrainedWord2VecModel(outputSummaryDir,
				corpusName, patternList);
		instances.addThruPipe(patternList.iterator());*/
		
		ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".featuredInsts"));
		instances.readObject(inInst);
		inInst.close();

//		System.out.println("Begin generate final summary");
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);
		
		int length = 0;
		boolean flag = false;
//		int numClusters = 25;
		
//		InstanceList[] groups = kmeans(instances, numClusters, proxy);
//		InstanceList[] groups = spectral(instances, numClusters, proxy);
		Object[] objs = seedBasedClustering(outputSummaryDir, corpusName,
				instances, categoryId, proxy, fvGenerator, sigma);
		Clustering predicted = (Clustering) objs[0];
		InstanceList[] groups = predicted.getClusters();
		HashMap<InstanceList, Integer> tmp = new HashMap<InstanceList, Integer>();
		for (InstanceList il : groups)
			tmp.put(il, il.size());
		HashMap rankedMap = RankMap.sortHashMapByValues(tmp, false);
		Set keys = rankedMap.keySet();
		Iterator iter = keys.iterator();
		Metric metric = new NormalizedDotProductMetric();
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
			String line = realization(p, graph);
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
	
	public void run_ActiveLearning(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, int parameter, MatlabProxy proxy) throws Exception {
		int topN = 10;
		double threshold = 0.7;
		int nIter = 5;
		
		InstanceList instances = new InstanceList(pipeLine);
		Metric metric = new NormalizedDotProductMetric();
		
		ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".featuredInsts"));
		instances.readObject(inInst);
		inInst.close();
		
		FeatureVector fv_0 = (FeatureVector) instances.get(0).getData();
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
		                        	
		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
		
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
			
		//realization
		ArrayList<String> tupleCandidates = tupleFusion(patternCluster);
		if(tupleCandidates == null)
			return;
/*		ArrayList<String> patternCandidates = patternFusion(patternCluster);	
		if(patternCandidates == null)
			return ;
		if (tupleCandidates.size() == 0 || patternCandidates.size() == 0) {
			System.out.println("tuple or pattern set is empty");
			return;
		}*/
		
		FeatureVectorGenerator fvGenerator = (FeatureVectorGenerator) pipeLine
				.getPipe(0);
		
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".smallWordMap"));
		HashMap<String, float[]> wordMap = (HashMap<String, float[]>)in.readObject();
		in.close();
		
/*		InstanceList patternInstances = new InstanceList(pipeLine);
		for(String pattern : patternCandidates){
			double[] vec_P = new double[300];
			int[] idx_P = new int[300];
			for(int a = 0; a <300; a++){
				idx_P[a] = a;
			}
			
			String[] tokPat = pattern.split(" ");
			for(String tokP : tokPat){
				tokP = fvGenerator.cleaning(tokP.toLowerCase());
				float[] wordVector_P = wordMap.get(tokP);
				if (wordVector_P == null)
					continue;
				for (int a = 0; a < 300; a++) {
					vec_P[a] += wordVector_P[a];
				}	
				
				float len = 0;
				for (int a = 0; a < 300; a++) {
					len += vec_P[a] * vec_P[a];
				}
				len = (float) Math.sqrt(len);
				for (int a = 0; a < 300; a++) {
					vec_P[a] /= len;
				}
	
			}
			
			FeatureVector fv = new FeatureVector(idx_P, vec_P);
			Instance patternInst = new Instance(fv, null, null, pattern);
			patternInstances.add(patternInst);
		}*/
		
/*		fv_0 = (FeatureVector) patternInstances.get(0).getData();
		dataMatrix = new double[fv_0.getIndices().length][patternInstances
		  		                        				.size()];
		
		for (int i = 0; i < patternInstances.size(); i++) {
			FeatureVector fv_i = (FeatureVector) patternInstances.get(i).getData();
		    double[] vals = fv_i.getValues();
		    for(int j=0; j< vals.length; j++)
		    dataMatrix[j][i] = vals[j];

		}
		 
		X0 = new double[patternInstances.size()][patternInstances.size()];
		for (int i = 0; i < patternInstances.size(); i++) {
			for(int j=0; j<patternInstances.size(); j++)
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
				for(int j=1; j<=patternInstances.size(); j++){
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
		
		
		rank_idx = processor.getNumericArray("rank_idx").getRealArray2D();  
		patternCluster = new InstanceList(pipeLine);
		for(int i=0; i<topN; i++){
			if(i < patternInstances.size())
				patternCluster.add(patternInstances.get((int)(rank_idx[i][0]-1)));				
			else{
				patternCluster.add(patternInstances.get((int)(rank_idx[patternInstances.size()-1][0]-1)));
			}
		}*/
		
		//construct tuple Instances
		InstanceList tupleInstances = new InstanceList(pipeLine);
		for(String tuple : tupleCandidates){

			String[] tokTup = tuple.split(" ");
			double[] vec_T = new double[300];
			int[] idx_T = new int[300];
			for(int a = 0; a <300; a++){
				idx_T[a] = a;
			}
			
			for(String tokT : tokTup){
				tokT = fvGenerator.cleaning(tokT.toLowerCase());
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
				
			FeatureVector fv = new FeatureVector(idx_T, vec_T);
			Instance tupleInst = new Instance(fv, null, null, tuple);
			tupleInstances.add(tupleInst);
		}
		
		//LGC semi supervised learning
		SemiSupervisedClustering semiClustering = new 
		LocalglobalConsistencySemiSupervisedClustering(
		new Noop(), patternCluster, metric, proxy, 30);
		
		Clustering clusters = semiClustering.cluster(tupleInstances);
		
		//begin to rank tuples in each clusters
		HashMap<Instance, Double> nbestMap_tuple = getNbestMap(outputSummaryDir,
				corpusName, tupleCandidates);
		LinkedHashMap rankedmap_t = RankMap.sortHashMapByValues(nbestMap_tuple, true);
		HashMap<Instance, Double> nbestMap_tuple_N = new HashMap<Instance, Double>();
		Set<Instance> ks = rankedmap_t.keySet();
		Iterator iks = ks.iterator();
		double rank = 1.0;
		while(iks.hasNext()){
			Instance inst = (Instance)iks.next();
			nbestMap_tuple_N.put(inst, 1/(rank++));
		}
		
		HashMap<Instance, Double> tupleScoreMap = new HashMap<Instance, Double>();
		InstanceList rankedInstanceList = new InstanceList(pipeLine);
		int[] labels = new int[tupleInstances.size()];
		
		int clusterIdx = 0;
		int labelIdx = 0;
		Iterator iter = patternCluster.iterator();
		while (iter.hasNext() && clusterIdx < patternCluster.size()) {
			Instance patternInst = (Instance) iter.next();
			String pattern= (String) patternInst.getSource();
		
			InstanceList tuplesIncluster = clusters.getCluster(clusterIdx);
			for(Instance tupleInst : tuplesIncluster){
				double coverageScore = 0.0;
				String tuple = (String)tupleInst.getSource();
				String[] tokTup = tuple.split(" ");
				String[] tokPat = pattern.split(" ");
				double[] vec_T = new double[300];
				int[] idx_T = new int[300];
				for(int a = 0; a <300; a++){
					idx_T[a] = a;
				}
				
				for(String tokT : tokTup){
					tokT = fvGenerator.cleaning(tokT.toLowerCase());
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
					
					for(String tokP : tokPat){
						tokP = fvGenerator.cleaning(tokP.toLowerCase());
						float[] wordVector_P = wordMap.get(tokP);				
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
				coverageScore = 1/(1+Math.exp(-coverageScore));
				
				double score = 0.7*coverageScore + 0.3*fluencyScore;
				tupleScoreMap.put(tupleInst, score);
			}
			
			HashMap sortedScores = RankMap.sortHashMapByValues(tupleScoreMap, false);
			Set<String> ts = sortedScores.keySet();
			Iterator its = ts.iterator();
			int j = 0;
			while(its.hasNext() && (j++) < tuplesIncluster.size()){
				Instance tupleInst = (Instance)its.next();
				rankedInstanceList.add(tupleInst);
				labels[labelIdx++] = clusterIdx;
			}
			
			clusterIdx++;
				
		}
	
		Clustering rankedClusters = new Clustering(rankedInstanceList,
				patternCluster.size(), labels);
		
		//generate final summary with simple greedy method
		double[][] matrix = new double[patternCluster.size()][rankedInstanceList.size()];
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
		
		out.close();
		
		//begin ILP
/*		LpSolve solver = LpSolve.makeLp(0, rankedClusters.getInstances().size());
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
				300);

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

		// Redundancy Constraints
		for (int i = 0; i < cs.length; i++) {
			InstanceList clusteri = cs[i];
			for (int m = 0; m < clusteri.size(); m++) {

				for (int j = i + 1; j < cs.length; j++) {
					InstanceList clusterj = cs[j];
					for (int n = 0; n < clusterj.size(); n++) {

						solver.strAddConstraint(
								buildStrVector(m, n, i, j, 
										rankedClusters, rankedInstanceList.size(), metric),
								LpSolve.LE, 0.9);

					}

				}

			}

		}

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
		out.close();*/
	
	}
	
	public void run_ActiveLearning2(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, int parameter, MatlabProxy proxy) throws Exception {
		//1. find representative patterns as seeds
		int topN = 10;
		double threshold = 0.7;
		int nIter = 5;
		InstanceList instances = new InstanceList(pipeLine);
		Metric metric = new NormalizedDotProductMetric();
		
		ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".featuredInsts"));
		instances.readObject(inInst);
		inInst.close();
		
		FeatureVector fv_0 = (FeatureVector) instances.get(0).getData();
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
		                        	
		MatlabTypeConverter processor = new MatlabTypeConverter(proxy);
		
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
			
	
		//2, clustering tuples with LGC semi supervised learning
		SemiSupervisedClustering semiClustering = new 
		LocalglobalConsistencySemiSupervisedClustering(
		new Noop(), patternCluster, metric, proxy, 30);
		Clustering clusters = semiClustering.cluster(instances);
		
		//3, generate final summary
		FeatureVectorGenerator fvGenerator = (FeatureVectorGenerator) pipeLine
				.getPipe(0);
		int length = 0;
		boolean flag = false;
		InstanceList[] groups = clusters.getClusters();
		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName);
		HashMap<InstanceList, Integer> tmp = new HashMap<InstanceList, Integer>();
		for (InstanceList il : groups)
			tmp.put(il, il.size());
		HashMap rankedMap = RankMap.sortHashMapByValues(tmp, false);
		Set keys = rankedMap.keySet();
		Iterator iter = keys.iterator();

		while (iter.hasNext()) {
			InstanceList il = (InstanceList) iter.next();
			ArrayList<String> lines = realization(outputSummaryDir, corpusName,
					il, proxy, fvGenerator);
			if (lines == null)
				continue;
			for (String line : lines) {
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
			}

			if (flag == true)
				break;
		}
		
		out.close();
	}
	
	public void run_ActiveLearning3(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine, int topN, MatlabProxy proxy) throws Exception {
		int parameter = 10;
		//1. find representative patterns as seeds
//		System.out.println("find representative patterns as seeds");
		double threshold = 0.7;
		int nIter = 7;
		InstanceList instances = new InstanceList(pipeLine);
		Metric metric = new NormalizedDotProductMetric();
		
		ObjectInputStream inInst = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".featuredInsts"));
		instances.readObject(inInst);
		inInst.close();
		
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
		
		InstanceList patternCluster = new InstanceList(pipeLine);
		ObjectInputStream inPatternCluster = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".patternCluster."+parameter));
		patternCluster.readObject(inPatternCluster);
		inPatternCluster.close();
			
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
		FeatureVectorGenerator fvGenerator = (FeatureVectorGenerator) pipeLine
				.getPipe(0);
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".smallWordMap"));
		HashMap<String, float[]> wordMap = (HashMap<String, float[]>)in.readObject();
		in.close();
		InstanceList[] groups = clusters.getClusters();
		
		InstanceList rankedInstanceList = new InstanceList(pipeLine);
		ArrayList<Integer> labelsList = new ArrayList<Integer>();
		int clusterIdx = 0;
		int labelIdx = 0;
		for(int i=0; i<groups.length; i++){
			Instance patternInst = patternCluster.get(i);
			String bestPattern= (String) patternInst.getSource().toString();
			InstanceList group = groups[i];
			ArrayList<String> tupleCandidates = tupleFusion(group);
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
					tokT = fvGenerator.cleaning(tokT.toLowerCase());
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
						tokP = fvGenerator.cleaning(tokP.toLowerCase());
						float[] wordVector_P = wordMap.get(tokP);
						
						tokT = fvGenerator.cleaning(tokT.toLowerCase());
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
