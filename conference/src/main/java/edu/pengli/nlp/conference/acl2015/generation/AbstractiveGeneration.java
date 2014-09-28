package edu.pengli.nlp.conference.acl2015.generation;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashSet;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

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


import edu.stanford.nlp.trees.EnglishGrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalRelation;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeGraphNode;
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

	public AbstractiveGeneration() {
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);
	}
	
	private Tree parseSentence(Instance sent){
			return null;
	}
	
	private Set<TreeGraphNode> getObjects(EnglishGrammaticalStructure eg,
			TreeGraphNode t) {
		Set<TreeGraphNode> objs = new HashSet<TreeGraphNode>();
		Collection<TypedDependency> tds = eg.typedDependenciesCollapsedTree();
		for (TypedDependency td : tds) {
			TreeGraphNode gov = td.gov();
			TreeGraphNode dep = td.dep();
			GrammaticalRelation gr = td.reln();
			if (gr.getShortName().equals("dobj") && gov.equals(t)) {
				objs.add(dep);
			}
		}
		return objs;
	}
	
	private Set<TreeGraphNode> getSubjects(EnglishGrammaticalStructure eg,
			TreeGraphNode t) {
		Set<TreeGraphNode> subs = new HashSet<TreeGraphNode>();
		Collection<TypedDependency> tds = eg.typedDependenciesCollapsedTree();
		for (TypedDependency td : tds) {
			TreeGraphNode gov = td.gov();
			TreeGraphNode dep = td.dep();
			GrammaticalRelation gr = td.reln();
			if (gr.toString().equals("nsubj") && gov.equals(t)) {
				if (!dep.toOneLineString().contains("that"))
					subs.add(dep);
			} else if (gr.toString().equals("rcmod") && dep.equals(t)) {
				subs.add(gov);
			} else if (gr.toString().equals("agent") && gov.equals(t)) {
				subs.add(dep);
			} else if (gr.toString().equals("dep") && dep.equals(t)) {
				subs.add(gov);
			} else if (gr.toString().equals("conj_and") && dep.equals(t)) {
				subs.add(gov);
			}

		}
		return subs;
	}
	
	private ArrayList<InformationItem> extractInformationItems(Tree parseTree){
		EnglishGrammaticalStructure eg = new EnglishGrammaticalStructure(
				parseTree);
		// step 1 collect possible predicates
		HashSet<TreeGraphNode> predicates = new HashSet<TreeGraphNode>();
		Collection<TypedDependency> tds = eg.typedDependenciesCollapsedTree();
		// parseTree.pennPrint();
		// Main.writeImage(parseTree, tds, "image.png", 3);
		for (TypedDependency td : tds) {
			TreeGraphNode gov = td.gov();
			// System.out.println(gov.toString());
			TreeGraphNode dep = td.dep();
			// System.out.println(dep.toString());
			GrammaticalRelation gr = td.reln();
			if (gr.toString().equals("nsubj") || gr.toString().equals("dobj")
					|| gr.toString().equals("xcomp")
					|| gr.toString().equals("agent")) {
				predicates.add(gov);
			}
		}

		ArrayList<InformationItem> tmpItems = new ArrayList<InformationItem>();

		for (TreeGraphNode p : predicates) {
			Set<TreeGraphNode> subjs = this.getSubjects(eg, p);
			if (subjs.size() == 0)
				continue;
			for (TreeGraphNode s : subjs) {
				Set<TreeGraphNode> deps = this.getObjects(eg, p);
				if (deps.size() != 0) {
					for (TreeGraphNode dep : deps) {
						InformationItem item = new InformationItem(s, p, dep);
						tmpItems.add(item);
					}
				} else {

					InformationItem item = new InformationItem(s, p, null);
					tmpItems.add(item);
				}

			}
		}

		ArrayList<InformationItem> items = new ArrayList<InformationItem>();
		ArrayList<TreeGraphNode> objs = new ArrayList<TreeGraphNode>();
		for (InformationItem item : tmpItems) {
			TreeGraphNode subject = item.getSubject();
			if (!predicates.contains(subject)) {
				if (item.getObject() != null)
					objs.add(item.getObject());
				items.add(item);
			}
		}

		for (InformationItem item : tmpItems) {
			TreeGraphNode subject = item.getSubject();
			if (predicates.contains(subject)) {
				for (TreeGraphNode it : objs) {
					InformationItem newItem = new InformationItem(it,
							item.getPredicate(), item.getObject());
					items.add(newItem);
				}

			}
		}
		return items;

	}
	
	private ArrayList<String> generate(Instance sent){
		Tree parsedTree = parseSentence(sent);
		EnglishGrammaticalStructure eg = new 
				EnglishGrammaticalStructure(parsedTree);

		Collection<TypedDependency> tds = eg.typedDependenciesCollapsedTree();
		HashSet<TreeGraphNode> set = new HashSet<TreeGraphNode>();
		for(TypedDependency td : tds){
			set.add(td.dep());
			set.add(td.gov());
		}
	
		DependencyGraph dg = new DependencyGraph(set.size());

		for (TypedDependency td : tds) {
			dg.addEdge(td);
		}
		SPhraseSpec newSent = nlgFactory.createClause();
		ArrayList<String> comSents = new ArrayList<String>();
		ArrayList<InformationItem> items = extractInformationItems(parsedTree);
		if (items.size() != 0)
			for (InformationItem item : items) {

				NPPhraseSpec subjectNp = generateNP(dg, item.getSubject());

				// System.out.println(lt.getRealiser().realiseSentence(subjectNp));

				newSent.setSubject(subjectNp);

				VPPhraseSpec vp = generateVP(dg, item.getPredicate(),
						item.getObject());

				newSent.setVerbPhrase(vp);

				String output = realiser.realiseSentence(newSent);

				comSents.add(output);
			}
		return comSents;
	}
	
	private NPPhraseSpec generateNP(DependencyGraph graph, TreeGraphNode head) {

		NPPhraseSpec np = nlgFactory.createNounPhrase();
		np.setHead(head.headWordNode().value());
		Stack<Integer> stack = new Stack();
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
				if (td.reln().toString().startsWith("prep")) {
					String prep = td.reln().toString().replace("prep_", "");
					TreeGraphNode obj = td.dep();
					PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
					if (np.getPostModifiers().size() != 0) {
						np.addPostModifier(ppp);
					} else
						np.setPostModifier(ppp);
					continue; // do not deep travel any more
				} else if (td.reln().toString().equals("nn")) {
					NPPhraseSpec tmp = generateNP(graph, td.dep());
					if (td.dep().index() < head.index())
						np.setPreModifier(tmp);
					else
						np.setPostModifier(tmp);
					continue;
				} else if (td.reln().toString().startsWith("conj")) {
					NPPhraseSpec tmp = generateNP(graph, td.dep());
					String conj = td.reln().toString().replace("conj_", "");
					np.addPostModifier(conj + " ");
					np.addPostModifier(tmp);
					continue;
				} else if (td.reln().toString().equals("det")) {
					TreeGraphNode det = td.dep();

					np.setSpecifier(det.value());
				} else if (td.reln().toString().equals("num")) {
					TreeGraphNode numMod = td.dep();
					np.addPreModifier(numMod.value());

				} else if (td.reln().toString().equals("amod")) {
					TreeGraphNode adjMod = td.dep();
					if (adjMod.index() < head.index())
						np.addPreModifier(adjMod.value());
					else
						np.addPostModifier(adjMod.value());
				} else
					continue; // this is ignore all the other children

				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}

		return np;
	}

	// follow the order to set vp
	private VPPhraseSpec generateVP(DependencyGraph graph,
			TreeGraphNode headVp, TreeGraphNode object) {

		VPPhraseSpec vp = nlgFactory.createVerbPhrase();
/*		String verbLemma = lemma(headVp.value());
		vp.setVerb(verbLemma);
		vp.setFeature(Feature.TENSE, Tense.PAST);*/

		if (object != null) {
			// set direct object
			NPPhraseSpec dirObjNp = generateNP(graph, object);
			vp.setObject(dirObjNp);

			// set indirect object from direct children
			Iterable<TypedDependency> iter = graph.adj(headVp.index());
			for (TypedDependency td : iter) {
				if (td.reln().toString().startsWith("iobj")) {
					NPPhraseSpec indirObjNp = generateNP(graph, td.dep());
					vp.setIndirectObject(indirObjNp);
					break;
				}
			}

		} else {
			// set verb complement
			Iterable<TypedDependency> iter = graph.adj(headVp.index());
			for (TypedDependency td : iter) {
				if (td.reln().toString().startsWith("ccomp")
						&& (td.dep().index() > headVp.index())) {
					String comp = generateClauseComplement(graph, td.dep());
					if (vp.getPostModifiers().size() != 0) {
						vp.addPostModifier(comp);
					} else
						vp.setPostModifier(comp);
				}
			}

		}

		// set prep and open clause complement if it has
		Stack<Integer> stack = new Stack();
		boolean[] marked = new boolean[graph.V()];
		int headIdx = headVp.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<TypedDependency> iter = graph.adj(s);
			for (TypedDependency td : iter) {
				int depIdx = td.dep().index();
				if (td.reln().toString().startsWith("prep")) {
					String prep = null;
					if (td.reln().toString().equals("prep")) {
						continue;
					}
					prep = td.reln().toString().replaceAll("prepc?_", ""); // dependency
																			// has
																			// prepc
					prep = prep.replaceAll("_", " "); // prep_out_of
					TreeGraphNode obj = td.dep();
					PPPhraseSpec ppp = generatePrepP(graph, prep, obj);
					if (ppp == null)
						continue;
					if (vp.getPostModifiers().size() != 0) {
						vp.addPostModifier(ppp);
					} else
						vp.setPostModifier(ppp);
				} else if (td.reln().toString().startsWith("xcomp")) {
					String comp = generateClauseComplement(graph, td.dep());
					if (vp.getPostModifiers().size() != 0) {
						vp.addPostModifier(comp);
					} else
						vp.setPostModifier(comp);

				} else
					continue;// this is ignore all the other children

				if (!marked[depIdx]) {
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}

		}
		return vp;
	}

	private PPPhraseSpec generatePrepP(DependencyGraph graph, String prep,
			TreeGraphNode np) {
		PPPhraseSpec ppp = nlgFactory.createPrepositionPhrase();
		ppp.setPreposition(prep);
		NPPhraseSpec tmp = generateNP(graph, np);
		if (tmp.equals(null))
			return null;
		String tmpnp = realiser.realiseSentence(tmp);
		tmpnp = tmpnp.replaceAll("[,|.]", ""); // because addmodificy can add,
												// and .
		ppp.setObject(tmpnp);
		return ppp;
	}

	private String generateClauseComplement(DependencyGraph graph,
			TreeGraphNode predicate) {

		Stack<Integer> stack = new Stack();
		boolean[] marked = new boolean[graph.V()];
		int headIdx = predicate.index();
		marked[headIdx] = true;
		stack.add(headIdx);
		TreeMap<Integer, TreeGraphNode> map = new TreeMap<Integer, TreeGraphNode>();
		map.put(predicate.index(), predicate);
		while (!stack.isEmpty()) {
			int s = stack.pop();
			Iterable<TypedDependency> iter = graph.adj(s);
			for (TypedDependency td : iter) {
				int depIdx = td.dep().index();
				if (!marked[depIdx]) {
					map.put(depIdx, td.dep());
					marked[depIdx] = true;
					stack.add(depIdx);
				}
			}
		}
		StringBuilder sb = new StringBuilder();
		Set<Integer> keys = map.keySet();
		for (Integer i : keys) {
			sb.append(map.get(i).value() + " ");
		}

		return sb.toString().trim();
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
		
		for(Instance sent : totalSentenceList){
			ArrayList<String> candidateSents = generate(sent);
		}
		
	}

}
