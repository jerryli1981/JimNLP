package edu.pengli.nlp.platform.algorithms.classify;

import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.Serializable;

import edu.pengli.nlp.platform.pipe.Pipe;
import edu.pengli.nlp.platform.types.InstanceList;


public abstract class Clusterer implements Serializable {
	
	Pipe instancePipe;
	
	/**
	 * Creates a new <code>Clusterer</code> instance.
	 *
	 * @param instancePipe Pipe that created the InstanceList to be
	 * clustered.
	 */
	public Clusterer(Pipe instancePipe) {
		this.instancePipe = instancePipe;
	}
	
	/** Return a clustering of an InstanceList */
	public abstract Clustering cluster (InstanceList trainingSet);

	public Pipe getPipe () { return instancePipe; }
	
	// SERIALIZATION

  private static final long serialVersionUID = 1;
  private static final int CURRENT_SERIAL_VERSION = 1;

  private void writeObject (ObjectOutputStream out) throws IOException {
    out.defaultWriteObject ();
    out.writeInt (CURRENT_SERIAL_VERSION);
  }

  private void readObject (ObjectInputStream in) throws IOException, ClassNotFoundException {
    in.defaultReadObject ();
    int version = in.readInt ();
  }	
}
