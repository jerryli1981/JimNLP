package edu.pengli.nlp.platform.types;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;




import edu.pengli.nlp.platform.pipe.Noop;
import edu.pengli.nlp.platform.pipe.Pipe;

public class InstanceList extends ArrayList<Instance> implements Serializable {

	Pipe pipe;

	Alphabet dataAlphabet;
	
	/**
	 * Construct an InstanceList having given capacity, with given default pipe.
	 * Typically Instances added to this InstanceList will have gone through the 
	 * pipe (for example using instanceList.addThruPipe); but this is not required.
	 * This InstanaceList will obtain its dataAlphabet and targetAlphabet from the pipe.
	 * It is required that all Instances in this InstanceList share these Alphabets. 
	 * @param pipe The default pipe used to process instances added via the addThruPipe methods.
	 * @param capacity The initial capacity of the list; will grow further as necessary.
	 */
	// XXX not very useful, should perhaps be removed
	public InstanceList (Pipe pipe, int capacity)
	{
		super(capacity);
		this.pipe = pipe;
	}

	/**
	 * Construct an InstanceList with initial capacity of 10, with given default pipe.
	 * Typically Instances added to this InstanceList will have gone through the 
	 * pipe (for example using instanceList.addThruPipe); but this is not required.
	 * This InstanaceList will obtain its dataAlphabet and targetAlphabet from the pipe.
	 * It is required that all Instances in this InstanceList share these Alphabets. 
	 * @param pipe The default pipe used to process instances added via the addThruPipe methods.
	 */
	public InstanceList (Pipe pipe)
	{
		this (pipe, 10);
	}

	
	/** Appends the instance to this list without passing the instance through
	 * the InstanceList's pipe.  
	 * The alphabets of this Instance must match the alphabets of this InstanceList.
	 * @return <code>true</code>
	 */
	public boolean add (Instance instance)
	{
		if (dataAlphabet == null)
			dataAlphabet = instance.getDataAlphabet();

		return super.add (instance);
	}

	public void addThruPipe(Iterator<Instance> ii) {
		Iterator<Instance> pipedInstanceIterator = pipe.newIteratorFrom(ii);
		while (pipedInstanceIterator.hasNext()) {
			Instance inst = pipedInstanceIterator.next();
			add(inst);
		}
	}
	
	public void setDataAlphabet(Alphabet dataAlphabet) {
		this.dataAlphabet = dataAlphabet;
	}
	

	public Alphabet getDataAlphabet() {
		if (dataAlphabet == null && pipe != null) {
			dataAlphabet = pipe.getDataAlphabet ();
		}
		return dataAlphabet;
	}
	

	private void writeObject(ObjectOutputStream out) throws IOException {
           out.writeObject(dataAlphabet);
		
	}

	private void readObject(ObjectInputStream in) throws IOException,
			ClassNotFoundException {
		dataAlphabet = (Alphabet) in.readObject();
	}


	public Pipe getPipe() {
		// TODO Auto-generated method stub
		return pipe;
	}

}
