package edu.pengli.nlp.platform.util;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.List;
import java.util.concurrent.Callable;

public class CallableTask implements Callable{
	
	Object cls;
	Method method;
	List<Object> args;
	
	public CallableTask(Object cls, Method method, List<Object> args){
		super();
		this.cls = cls;
		this.method = method;
		this.args = args;
	}
	
	public Object call() throws Exception{
		Object rs = null;
		if(args == null){
			rs = method.invoke(cls);
		}else{
			rs = method.invoke(cls, args.toArray());
		}
		return rs;
	}


}
