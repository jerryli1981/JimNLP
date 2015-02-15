package edu.pengli.nlp.conference.acl2015.generation;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

import edu.pengli.nlp.conference.acl2015.types.Argument;
import edu.pengli.nlp.conference.acl2015.types.Pattern;
import edu.pengli.nlp.conference.acl2015.types.Predicate;
import edu.stanford.nlp.ling.IndexedWord;
import edu.stanford.nlp.semgraph.SemanticGraph;
import edu.stanford.nlp.semgraph.SemanticGraphEdge;
import edu.stanford.nlp.trees.GrammaticalRelation;
import simplenlg.framework.NLGFactory;
import simplenlg.lexicon.Lexicon;
import simplenlg.phrasespec.NPPhraseSpec;
import simplenlg.phrasespec.PPPhraseSpec;
import simplenlg.phrasespec.SPhraseSpec;
import simplenlg.phrasespec.VPPhraseSpec;
import simplenlg.realiser.english.Realiser;

public class NLGGenerator {
	
	NLGFactory nlgFactory;
	Realiser realiser;
	
	public NLGGenerator(){
		Lexicon lexicon = Lexicon.getDefaultLexicon();
		nlgFactory = new NLGFactory(lexicon);
		realiser = new Realiser(lexicon);	
	}
	
	
	
	public String realization(Pattern p, SemanticGraph graph) {

		SPhraseSpec newSent = nlgFactory.createClause();
		IndexedWord arg1iw = p.getArg1().getHead();
		NPPhraseSpec subjectNp = generateNP(graph, arg1iw);
		newSent.setSubject(subjectNp);

		VPPhraseSpec vp = generateVP(graph, p.getRel(), p.getArg2());
		newSent.setVerbPhrase(vp);

		String output = realiser.realiseSentence(newSent);

		return output;

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

}
