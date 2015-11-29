package edu.pengli.nlp.platform.pipe;

import java.io.Serializable;
import java.util.Iterator;

import edu.pengli.nlp.platform.types.Alphabet;
import edu.pengli.nlp.platform.types.Instance;

public abstract class Pipe implements Serializable{

	Alphabet dataAlphabet = null;
	Alphabet targetAlphabet = null;

	public Pipe ()
	{
		this (null, null);
	}

	public Pipe (Alphabet dataDict, Alphabet targetDict)
	{
		this.dataAlphabet = dataDict;
		this.targetAlphabet = targetDict;
	}

	public Iterator<Instance> newIteratorFrom(Iterator<Instance> ii) {
		return new SimplePipeInstanceIterator(ii);
	}

	protected abstract Instance pipe(Instance inst);

	public Alphabet getDataAlphabet() {
		return dataAlphabet;
	}

	public Alphabet getTargetAlphabet() {
		return targetAlphabet;
	}

	private class SimplePipeInstanceIterator implements Iterator<Instance> {
		Iterator<Instance> ii;

		public SimplePipeInstanceIterator(Iterator<Instance> ii) {
			this.ii = ii;
		}

		public boolean hasNext() {
			return ii.hasNext();
		}

		public Instance next() {
			Instance inst = ii.next();
			return pipe(inst);
		}

		public void remove() {
			throw new IllegalStateException("Not supported");

		}

	}
}
