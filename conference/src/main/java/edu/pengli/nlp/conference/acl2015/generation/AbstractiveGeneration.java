package edu.pengli.nlp.conference.acl2015.generation;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

import org.json.JSONException;

import edu.pengli.nlp.conference.acl2015.pipe.FramenetTagger;
import edu.pengli.nlp.conference.acl2015.pipe.FreebaseTagger;
import edu.pengli.nlp.conference.acl2015.pipe.HeadExtractor;
import edu.pengli.nlp.conference.acl2015.pipe.DBpediaTagger;
import edu.pengli.nlp.conference.acl2015.pipe.WordnetTagger;
import edu.pengli.nlp.conference.acl2015.types.InformationItem;
import edu.pengli.nlp.conference.acl2015.types.Predicate;
import edu.pengli.nlp.conference.acl2015.types.Tuple;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.pengli.nlp.platform.util.FileOperation;
import edu.pengli.nlp.platform.util.RankMap;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphCoreAnnotations.BasicDependenciesAnnotation;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.CollinsHeadFinder;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.HeadFinder;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.util.CoreMap;
import scala.collection.Seq;
import scala.collection.Iterator;
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

	DBpediaTagger dbpediaTagger;

	FreebaseTagger freebaseTagger;

	WordnetTagger wordnetTagger;
	
	FramenetTagger framenetTagger;

	public AbstractiveGeneration() {

		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);

		dbpediaTagger = new DBpediaTagger();
		wordnetTagger = new WordnetTagger();
		freebaseTagger = new FreebaseTagger();
		framenetTagger = new FramenetTagger();

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

	private ArrayList<String> generate(SemanticGraph graph) {

		SPhraseSpec newSent = nlgFactory.createClause();
		ArrayList<String> comSents = new ArrayList<String>();

		ArrayList<InformationItem> items = extractInformationItems(graph);

		if (items.size() != 0)
			for (InformationItem item : items) {

				System.out.println("Information Item is: " + item.toString());

				NPPhraseSpec subjectNp = generateNP(graph, item.getSubject());

				newSent.setSubject(subjectNp);

				VPPhraseSpec vp = generateVP(graph, item.getPredicate(),
						item.getObject());

				newSent.setVerbPhrase(vp);

				String output = realiser.realiseSentence(newSent);

				System.out.println("Generated sent is: " + output);

				comSents.add(output);
			}
		return comSents;
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
			throws IOException, ClassNotFoundException, JSONException {

		ObjectInputStream in = new ObjectInputStream(new FileInputStream(
				outputSummaryDir + "/" + corpusName + ".ser"));

		corpus.readObject(in);

		PrintWriter out = FileOperation.getPrintWriter(new File(
				outputSummaryDir), corpusName + ".patterns");
		
		HashSet<String> tupleMentions = new HashSet<String>();

		for (Instance doc : corpus) {

			HashMap<CoreMap, ArrayList<Tuple>> map = (HashMap<CoreMap, ArrayList<Tuple>>) doc
					.getData();

			for (CoreMap sent : map.keySet()) {
								
//				out.println(sent.toString());
				ArrayList<Tuple> tuples = map.get(sent);
				for (Tuple t : tuples) {
					if (t.gerRel().toString().equals("said"))
						continue;
					
					edu.pengli.nlp.conference.acl2015.types.Argument arg1Head = headExtractor
							.extract(t.getArg1(), sent);

					edu.pengli.nlp.conference.acl2015.types.Argument arg2Head = headExtractor
							.extract(t.getArg2(), sent);

					

					if (arg1Head != null && arg2Head != null) {
						t.setArg1(arg1Head);
						t.setArg2(arg2Head);

					
						boolean ar1Prp = false;
						if(arg1Head.get(0).tag().equals("PRP"))
							ar1Prp = true;
						
						boolean ar2Prp = false;
						if(arg2Head.get(0).tag().equals("PRP"))
							ar2Prp = true;
						
						
						if((!arg1Head.get(0).ner().equals("O") || ar1Prp) 
								&& (!arg2Head.get(0).ner().equals("O") || ar2Prp))
							continue;
						
						if(arg1Head.get(0).ner().equals("O") && arg2Head.get(0).ner().equals("O")){
							tupleMentions.add(t.getArg1()+" "+t.gerRel()+" "+t.getArg2());
						}
/*						if (arg1Head.get(0).ner().equals("O")) {
							String arg1Mention = arg1Head.get(0).lemma();
							if (arg1Head.get(0).tag().startsWith("NN")) {
								out.println(arg1Mention + "/"
										+ arg1Head.get(0).tag() + "->"
										+ arg1Head.get(0).ner());
								ArrayList<String> label1s = wordnetTagger
										.getNounTypes(arg1Mention);
								out.println(arg1Mention + "<-wordNet->"
										+ label1s);
								String label1ByDB = dbpediaTagger
										.NameEntityRecognition(arg1Mention);
								System.out.println(arg1Mention + "<-dbpedia->"
										+ label1ByDB);

								ArrayList<String> freeLabels = freebaseTagger
										.search(arg1Mention);
								out.println(arg1Mention + "<-freebase->"
										+ freeLabels);

							}
						}

						if (arg2Head.get(0).ner().equals("O")) {
							String arg2Mention = arg2Head.get(0).lemma();
							if (arg2Head.get(0).tag().startsWith("NN")) {
								out.println(arg2Mention + "/"
									+ arg2Head.get(0).tag() + "->"
										+ arg2Head.get(0).ner());
								ArrayList<String> label2s = wordnetTagger
										.getNounTypes(arg2Mention);
								out
										.println(arg2Mention + "<-wordnet->" + label2s);
								String label2ByDB = dbpediaTagger
										.NameEntityRecognition(arg2Mention);
								System.out.println(arg2Mention + "<-dbpedia->"
										+ label2ByDB);

								ArrayList<String> freeLabels = freebaseTagger
										.search(arg2Mention);
								out.println(arg2Mention + "<-freebase->"
										+ freeLabels);
							}

						}*/

						//out.println(t);
						//framenetTagger
						out.println(t);
					}		
				}
			}
		}

		out.close();

	}

	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName, PipeLine pipeLine) throws IOException,
			ClassNotFoundException, JSONException {

		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName);

		InstanceList docs = new InstanceList(pipeLine);

		/*
		 * docs.addThruPipe(fIter); ObjectOutputStream out = new
		 * ObjectOutputStream(new FileOutputStream( outputSummaryDir + "/" +
		 * corpusName + ".ser")); docs.writeObject(out); out.close();
		 */

		System.out.println("Begin generate patterns");
		HeadExtractor headExtractor = new HeadExtractor();
		generatePatterns(outputSummaryDir, corpusName, docs, headExtractor);

	}

}
