
package edu.pengli.nlp.platform.pipe;


import java.io.Serializable;

import edu.pengli.nlp.platform.types.Instance;


public class Noop extends Pipe implements Serializable
{
	public Noop ()
	{
	}


	public Instance pipe (Instance carrier)
	{
		return carrier;
	}



}
