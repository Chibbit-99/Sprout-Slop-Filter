// Rejected experiment: validation improved but the selected head regressed on
// protected held-out/unfamiliar examples. DO NOT ship or tune on those results.
import fs from 'node:fs';
import {sprout} from './lib/sprout.mjs';
const read = file => JSON.parse(fs.readFileSync(new URL(file,import.meta.url),'utf8'));
const loaded = sprout.load_model('model.json',{storage:()=>fs.readFileSync('model.json','utf8')});
if(!loaded.success)throw Error(loaded.error);
const model=loaded.model;
const parts=sprout.split_dataset(read('../training.json'),{seed:1});
const original=sprout.split_dataset(read('task_baseline_training.json'),{seed:1});
const additional=read('task_supplement.json');
const lex=model.options.lexicon;
const features=text=>{
 const words=sprout.tokenise(sprout.normalise_text(text)).slice(0,128);
 const cats=words.map(w=>sprout.lexicon_entry_for(w,lex)?.[0]||'');
 const content=cats.filter(c=>/^n\d+/.test(c));
 const vague=new Set(['some','various','miscellaneous','stuff','things','thing','matters','something','anything','everything','whatever','general','usual','unspecified','somehow','vague']);
 const weak=new Set(['do','make','improve','enhance','handle','address','review','work','look','think','discuss','fix']);
 const qualifiers=new Set(['before','after','by','for','at','in','on','from','to','with','into','under','behind','through','between']);
 const numbers=words.filter(w=>/\d/.test(w));
 const nounQuant=Math.min(3,content.length)/3;
 const genericCount=Math.min(3,words.filter(w=>vague.has(w)).length)/3;
 const feat=[
  Math.min(words.length,12)/12,
  +!!numbers.length,
  +numbers.some(w=>/\p{L}/u.test(w)),
  +(words.length>0 && /^\p{L}/u.test(words[0]) && words.length>2),
  +weak.has(words[0]),
  +words.some(w=>qualifiers.has(w)),
  +words.some(w=>['before','after','by','until','on','at'].includes(w)),
  nounQuant,
  +content.some(c=>['n06','n05','n04','n13','n27','n38'].includes(c)),
  +content.some(c=>['n07','n09','n10','n26','n28'].includes(c)),
  genericCount,
  +(genericCount>0 && weak.has(words[0])),
  +words.some(w=>['situation','project','work','task','tasks','process','data','issues','value'].includes(w)),
  +words.some(w=>['and','or','etc'].includes(w)),
  +words.some(w=>/[^\x00-\x7f]/.test(w)),
  +words.some(w=>/\w['’]\w/u.test(w)),
  Math.min(3,words.filter(w=>!Object.hasOwn(lex,w) && !model.vocabulary.includes(w)).length)/3,
  +words.some(w=>['?', '!'].includes(w)),
  +words.some(w=>w.length>=12),
  +words.some(w=>['the','a','an'].includes(w)),
  +words.some(w=>['specific','exact','named','which','how','why'].includes(w)),
  +words.some(w=>['later','eventually','soon','sometime'].includes(w)),
  +words.some(w=>['not','never','cannot','without'].includes(w)),
  +words.some(w=>['please','could','can','would'].includes(w)),
 ];
 return feat;
};
const rows=parts.trainSet.concat(additional).map((r,i)=>({text:r.text,x:features(r.text),z:Math.log(sprout.make_prediction_for_training(r.text,model).score/(1-sprout.make_prediction_for_training(r.text,model).score)),y:+(r.label==='slop'),group:i<parts.trainSet.length?0:1}));
const sigmoid=z=>1/(1+Math.exp(-z));
const logit=(r,w)=>r.z+r.x.reduce((s,x,i)=>s+x*w[i],0);
const train=(lambda,extraWeight)=>{
 let w=new Array(rows[0].x.length).fill(0), step=.4;
 for(let epoch=0;epoch<320;epoch++){
  const grad=new Array(w.length).fill(0);let norm=0;
  for(const r of rows){const weight=r.group?extraWeight:1; const e=(sigmoid(logit(r,w))-r.y)*weight; norm+=weight;for(let i=0;i<w.length;i++)grad[i]+=e*r.x[i]}
  for(let i=0;i<w.length;i++){w[i]-=step*(grad[i]/norm+lambda*w[i]);}step*=.991;
 }
 return w;
};
const score=(modelRows,w)=>{let fp=0,fn=0,loss=0;for(let r of modelRows){let p=sprout.make_prediction_for_training(r.text,model).score;let z=Math.log(p/(1-p));let x=features(r.text);let next=sigmoid(z+x.reduce((s,v,i)=>s+v*w[i],0));let label=r.label==='slop';if(next>=.5&&!label)fp++;if(next<.5&&label)fn++;loss+=-Number(label)*Math.log(Math.max(1e-8,next))-(1-Number(label))*Math.log(Math.max(1e-8,1-next));}return `${modelRows.length-fp-fn}/${modelRows.length} FP${fp} FN${fn} loss${(loss/modelRows.length).toFixed(3)}`};
const candidates=[];for(const lambda of [.01,.05,.2,1,4])for(const extra of [0,.1,.5,1]){let w=train(lambda,extra);let originalVal=score(original.validationSet,w), expandedVal=score(parts.validationSet,w);console.log({lambda,extra,originalVal,expandedVal,max:Math.max(...w.map(Math.abs)).toFixed(3)});candidates.push({lambda,extra,w,originalVal,expandedVal})}
console.log('base:',score(original.validationSet,new Array(rows[0].x.length).fill(0)),score(parts.validationSet,new Array(rows[0].x.length).fill(0)));
// Post-selection diagnostic only. Candidate was selected using the original
// validation set above; the tests/probes below must not tune its coefficients.
const selected=candidates.find(c=>c.lambda===.01&&c.extra===0);
for(const [name,rs] of [['originalHeldout',original.testSet],['expandedHeldout',parts.testSet],...Object.entries(read('unfamiliar_cases.json').groups),['helpdeskHard',read('other_contexts.json').helpdesk.stress],['agendaHard',read('other_contexts.json').meeting_agendas.stress]]){
 console.log(name,'base',score(rs,new Array(selected.w.length).fill(0)),'residual',score(rs,selected.w));
}
console.log('selected weights:',selected.w.map(n=>n.toFixed(3)).join(','));
