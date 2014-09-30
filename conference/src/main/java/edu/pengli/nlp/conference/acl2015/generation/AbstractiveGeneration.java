package edu.pengli.nlp.conference.acl2015.generation;

import java.io.StringReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

import edu.pengli.nlp.conference.acl2015.pipe.CharSequenceExtractContent;
import edu.pengli.nlp.conference.acl2015.types.InformationItem;
import edu.pengli.nlp.conference.acl2015.types.NewsCorpus;
import edu.pengli.nlp.platform.pipe.Input2CharSequence;
import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.PipeLine;
import edu.pengli.nlp.platform.pipe.iterator.OneInstancePerFileIterator;
import edu.pengli.nlp.platform.types.DependencyGraph;
import edu.pengli.nlp.platform.types.Instance;
import edu.pengli.nlp.platform.types.InstanceList;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.process.Tokenizer;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeGraphNode;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
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
	LexicalizedParser lp;
	TreebankLanguagePack tlp;
	GrammaticalStructureFactory gsf;

	public AbstractiveGeneration(LexicalizedParser lp) {
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);
		this.lp = lp;
		tlp = new PennTreebankLanguagePack();
		gsf = tlp.grammaticalStructureFactory();
	}

	private Tree parseSentence(Instance sent) {

		// the first way
		/*
		 * TokenizerFactory<CoreLabel> tokenizerFactory = PTBTokenizer.factory(
		 * new CoreLabelTokenFactory(), ""); Tokenizer<CoreLabel> tok =
		 * tokenizerFactory .getTokenizer(new StringReader((String)
		 * sent.getSource())); List<CoreLabel> rawWords2 = tok.tokenize();
		 * 
		 * return lp.apply(rawWords2);
		 */

		// //////////////////

		// the second way is better than first way in performance
		Tokenizer<? extends HasWord> toke = tlp.getTokenizerFactory()
				.getTokenizer(new StringReader((String) sent.getSource()));
		List<? extends HasWord> sentence = toke.tokenize();
		return lp.parse(sentence);

	}

	/*
	 * current implementation just cove direct object and prep object, subject
	 * and predicate are necessary. object could be empty.
	 */
	private ArrayList<InformationItem> extractInformationItems(Tree parseTree,
			Collection<TypedDependency> tds) {

		DependencyGraph graph = new DependencyGraph(parseTree.size());
		for (TypedDependency td : tds) {
			graph.addEdge(td);
		}

		HashSet<TreeGraphNode> predicates = new HashSet<TreeGraphNode>();
		TreeGraphNode subjectIncase = null;
		for (TypedDependency td : tds) {
			TreeGraphNode gov = td.gov();
			GrammaticalRelation gr = td.reln();
			if (gr.toString().equals("nsubj")
					|| gr.toString().equals("dobj")
					|| (gr.toString().equals("prep") && gov.parent()
							.nodeString().startsWith("VB"))) {
				predicates.add(gov);
			}

			if (gr.toString().equals("nsubj")) {
				Iterable<TypedDependency> iter = graph.adj(td.gov().index());
				for (TypedDependency child : iter) {
					GrammaticalRelation dgr = child.reln();
					if (dgr.toString().equals("dobj")
							|| (dgr.toString().equals("prep") && child.gov()
									.parent().nodeString().startsWith("VB"))) {
						subjectIncase = td.dep();
					}

				}

			}
		}

		ArrayList<InformationItem> possibleItems = new ArrayList<InformationItem>();

		for (TreeGraphNode p : predicates) {

			boolean subjectExist = false;
			boolean directObjectExist = false;
			boolean prepObjectExist = false;
			TreeGraphNode subject = null;
			TreeGraphNode directObject = null;
			TreeGraphNode prep = null;
			TreeGraphNode prepObject = null;

			for (TypedDependency td : tds) {
				TreeGraphNode gov = td.gov();
				if (!gov.equals(p))
					continue;
				TreeGraphNode dep = td.dep();
				GrammaticalRelation gr = td.reln();
				if (gr.toString().equals("nsubj")) {
					subjectExist = true;
					subject = dep;
				}
				if (gr.getShortName().equals("dobj")) {
					directObjectExist = true;
					directObject = dep;
				}
				if (gr.toString().equals("prep")
						&& gov.parent().nodeString().startsWith("VB")) {

					for (TypedDependency pair : tds) {
						TreeGraphNode g = pair.gov();
						if (dep.equals(g)
								&& pair.reln().toString().equals("pobj")) {
							prepObjectExist = true;
							prep = g;
							prepObject = pair.dep();
						}

					}
				}
			}

			if (subjectIncase == null)
				continue;

			if (subjectExist == false && directObjectExist == true
					&& prepObjectExist == false) {
				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
				obj.add(directObject);
				possibleItems.add(new InformationItem(subjectIncase, p, obj));

			} else if (subjectExist == false && directObjectExist == false
					&& prepObjectExist == true) {

				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(subjectIncase, p, obj));

			} else if (subjectExist == true && directObjectExist == false
					&& prepObjectExist == false) {

				possibleItems.add(new InformationItem(subject, p, null));

			} else if (subjectExist == true && directObjectExist == true
					&& prepObjectExist == false) {
				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
				obj.add(directObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			} else if (subjectExist == true && directObjectExist == false
					&& prepObjectExist == true) {

				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			} else if (subjectExist == true && directObjectExist == true
					&& prepObjectExist == true) {
				// One Amish man craned his head out a buggy window
				ArrayList<TreeGraphNode> obj = new ArrayList<TreeGraphNode>();
				obj.add(directObject);
				obj.add(prep);
				obj.add(prepObject);
				possibleItems.add(new InformationItem(subject, p, obj));
			}
		}

		return possibleItems;

	}

	private ArrayList<String> generate(Instance sent) {

//		sent.setSource("The suspect apparently called his wife from a cell phone shortly before the shooting began, saying he was acting out in revenge for something that happened 20 years ago, Miller said.");
		Tree parsedTree = parseSentence(sent);
		GrammaticalStructure gs = gsf.newGrammaticalStructure(parsedTree);
		Collection<TypedDependency> tds = gs.typedDependencies();

		DependencyGraph dg = new DependencyGraph(parsedTree.size());

		for (TypedDependency td : tds) {
			dg.addEdge(td);
		}

		SPhraseSpec newSent = nlgFactory.createClause();
		ArrayList<String> comSents = new ArrayList<String>();
		System.out.println("Original Sent is: " + sent.getSource());
		ArrayList<InformationItem> items = extractInformationItems(parsedTree,
				tds);

		if (items.size() != 0)
			for (InformationItem item : items) {

				System.out.println("Information Item is: " + item.toString());

				NPPhraseSpec subjectNp = generateNP(tds, item.getSubject(), dg);

				newSent.setSubject(subjectNp);

				VPPhraseSpec vp = generateVP(tds, item.getPredicate(),
						item.getObject(), dg);

				newSent.setVerbPhrase(vp);

				String output = realiser.realiseSentence(newSent);

				System.out.println("Generated sent is: " + output);

				comSents.add(output);
			}
		return comSents;
	}

	// search the tree recursively
	private TreeGraphNode searchObjforPrep(DependencyGraph graph,
			TreeGraphNode prepNode) {

		TreeGraphNode obj = null;
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.V()];
		int headIdx = prepNode.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		boolean stop =false;
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<TypedDependency> iter = graph.adj(s);
			
			for (TypedDependency td : iter) {
				GrammaticalRelation dgr = td.reln();
				if (dgr.toString().endsWith("obj") || dgr.toString().endsWith("pcomp")) {
					obj = td.dep();
					stop = true;
				}
				int depIdx = td.dep().index();
				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
			if(stop == true)
				break;
			
		}

		return obj;
	}

	private NPPhraseSpec generateNP(Collection<TypedDependency> tds,
			TreeGraphNode head, DependencyGraph graph) {

		NPPhraseSpec np = nlgFactory.createNounPhrase();
		np.setHead(head.headWordNode().value());
		Stack<Integer> stack = new Stack<Integer>();
		boolean[] marked = new boolean[graph.V()];
		int headIdx = head.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<TypedDependency> iter = graph.adj(s);
			for (TypedDependency td : iter) {
				if (td.gov().index() == td.dep().index())
					continue; // prevent infitive recusion

				int depIdx = td.dep().index();

				if (td.reln().toString().equals("prep")) {
					String prep = td.dep().nodeString();
					TreeGraphNode obj = searchObjforPrep(graph, td.dep());
					if (obj != null) {
						PPPhraseSpec ppp = generatePrepP(tds, prep, obj, graph);
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(ppp);
						} else
							np.setPostModifier(ppp);
					}

					continue; // do not deep travel any more

				} else if (td.reln().toString().equals("nn")) {
					NPPhraseSpec nounModifier = generateNP(tds, td.dep(), graph);
					if (td.dep().index() < head.index()) {
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

				} else if (td.reln().toString().equals("conj")) {

					Iterable<TypedDependency> children = graph.adj(td.gov()
							.index());
					TreeGraphNode cc = null;
					for (TypedDependency child : children) {
						GrammaticalRelation dgr = child.reln();
						if (dgr.toString().equals("cc")) {
							cc = child.dep();
						}
					}
					NPPhraseSpec nounModifier = generateNP(tds, td.dep(), graph);
					if(cc != null){
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(cc.nodeString());
							np.addPostModifier(nounModifier);
						} else{
							np.setPostModifier(cc.nodeString());
							np.addPostModifier(nounModifier);
						}	
					}else{
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(nounModifier);
						} else{
							np.addPostModifier(nounModifier);
						}	
					}

					continue;
				} else if (td.reln().toString().equals("det") || td.reln().toString().equals("poss")) {
					TreeGraphNode det = td.dep();
					np.setSpecifier(det.value());
				} else if (td.reln().toString().equals("num")) {

					TreeGraphNode numModifier = td.dep();
					np.setSpecifier(numModifier.value());

				} else if (td.reln().toString().equals("amod")) {
					TreeGraphNode adjMod = td.dep();
					if (adjMod.index() < head.index()) {
						if (np.getPreModifiers().size() != 0) {
							np.addPreModifier(adjMod.nodeString());
						} else
							np.setPreModifier(adjMod.nodeString());

					} else {
						if (np.getPostModifiers().size() != 0) {
							np.addPostModifier(adjMod.nodeString());
						} else
							np.setPostModifier(adjMod.nodeString());
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

	private VPPhraseSpec generateVP(Collection<TypedDependency> tds,
			TreeGraphNode headVp, ArrayList<TreeGraphNode> object,
			DependencyGraph graph) {

		VPPhraseSpec vp = nlgFactory.createVerbPhrase();
		vp.setHead(headVp.nodeString());
		// set aux of the headVerb
		Iterable<TypedDependency> children = graph.adj(headVp.index());
		for (TypedDependency td : children) {
			if (td.reln().toString().equals("aux")) {
				vp.setPreModifier(td.dep().nodeString());
				break;
			}
		}

		// set object
		if (object != null) {

			if (object.size() == 1) {
				// set direct object
				NPPhraseSpec dirObjNp = generateNP(tds, object.get(0), graph);
				vp.setObject(dirObjNp);
			}

			if (object.size() == 2) {
				// set prep object from direct children
				String prep = object.get(0).nodeString();
				TreeGraphNode obj = searchObjforPrep(graph, object.get(0));
				if (obj == null) {
					System.out
							.println("Prep Obj is NULL in generate VP. Program is stopped");
					System.exit(0);
				}
				PPPhraseSpec ppp = generatePrepP(tds, prep, obj, graph);
				vp.setObject(ppp);

			}

			if (object.size() == 3) {

				// set direct and prep object
				NPPhraseSpec dirObjNp = generateNP(tds, object.get(0), graph);
				vp.setObject(dirObjNp);

				String prep = object.get(1).nodeString();
				TreeGraphNode obj = searchObjforPrep(graph, object.get(1));
				if (obj == null) {
					System.out
							.println("Prep Obj is NULL in generate VP when obj size is 3. Program is stopped");
					System.exit(0);
				}

				PPPhraseSpec ppp = generatePrepP(tds, prep, obj, graph);
				vp.setPostModifier(ppp);

			}

		}
		return vp;
	}

	private PPPhraseSpec generatePrepP(Collection<TypedDependency> tds,
			String prep, TreeGraphNode np, DependencyGraph graph) {
		PPPhraseSpec ppp = nlgFactory.createPrepositionPhrase();
		ppp.setPreposition(prep);
		NPPhraseSpec npp = generateNP(tds, np, graph);
		ppp.setObject(npp);
		return ppp;
	}

	public void run(String inputCorpusDir, String outputSummaryDir,
			String corpusName) {

		OneInstancePerFileIterator fIter = new OneInstancePerFileIterator(
				inputCorpusDir + "/" + corpusName);

		PipeLine pipeLine = new PipeLine();
		pipeLine.addPipe(new Input2CharSequence("UTF-8"));
		pipeLine.addPipe(new CharSequenceExtractContent(
				"<TEXT>[\\p{Graph}\\p{Space}]*</TEXT>"));
		NewsCorpus corpus = new NewsCorpus(fIter, pipeLine);

		pipeLine = new PipeLine();
		pipeLine.addPipe(new Noop());
		NewsCorpus docs = new NewsCorpus(corpus, pipeLine);

		InstanceList totalSentenceList = new InstanceList(null);
		for (Instance doc : docs) {
			InstanceList sentsList = (InstanceList) doc.getSource();
			for (Instance inst : sentsList) {
				Instance sent = new Instance(inst.getSource(), null,
						inst.getName(), inst.getSource());
				totalSentenceList.add(sent);
			}
		}

		for (Instance sent : totalSentenceList) {
			ArrayList<String> candidateSents = generate(sent);

		}

	}

}
