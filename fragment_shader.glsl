#version 330 core
//#define SHADERTOY

#ifndef SHADERTOY
out vec4 fragColor;
in vec4 fragPosition;  // Variable d'entrée qui récupère la position du vertex shader

// Uniforms pour les données des doigts
uniform vec3 finger_positions[5]; // Positions 3D des bouts des doigts
uniform float finger_openings[5]; // Ouvertures des doigts
uniform float finger_speeds[5];   // Vitesses des doigts
uniform float pt0_speed;          // Vitesse du point 0
uniform float time;

vec2 iResolution = vec2(800,600);
float iTime;

void mainImage( out vec4 fragColor, in vec2 fragCoord );

void main() {
    // Exemple : Couleur changeante selon la vitesse du point 0
  //  float intensity = clamp(pt0_speed * 10.0, 0.0, 1.0);
  //  fragColor = vec4(intensity, 0.5, 1.0 - intensity, 1.0);
    iTime = time;
    vec2 fragCoord = (.5*fragPosition.xy+.5)*iResolution; ///2.0 - 1.0;
    mainImage(fragColor, fragCoord);
}

#endif


#define FAR 80.

mat2 rot(float a) {
    return mat2(cos(a), sin(a), -sin(a), cos(a));
}

float smin(float a, float b, float k){
    float h = clamp(.5+.5*(b-a)/k,0.,1.);
    return mix(b,a,h)-k*h*(1.-h);
}

// gyroid noise
float fbm_g(vec3 p) {
    float r=0.,a=.45;
    for(;a>.03;a*=.5)p+=r+=a*abs(dot(sin(p/a),cos(1.4*p.yzx/a)));
    return r;
}

float WaveletNoise(vec3 p, float z, float k) {
    float d=0.,s=1.,m=0., a0;
    mat2 mr = rot(.95);
    for(float i=0.; i<5.; i++) {
        vec3 q = p*s, g = fract(floor(q)*vec3(123.34,233.53,314.15));
    	g += dot(g, g + 23.234);
        q = fract(q)-.5;
        q.xy *= rot(10.*dot(g,g));
        d += sin(q.x*10.+z)*smoothstep(.25, .0, dot(q,q))/s;
        p.yx *= mr;
        p.yz *= mr;
        m += 1./s;
        s *= k; 
    }
    return d/m;
}

float sdCapsule( vec3 p, vec3 a, vec3 b, float r) {
    vec3 pa = p - a, ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

float sdEllipsoid( vec3 p, vec3 r )
{
  float k0 = length(p/r);
  float k1 = length(p/(r*r));
  return k0*(k0-1.0)/k1;
}

float sdRoundBox( vec3 p, vec3 b, float r ) {
  vec3 q = abs(p) - b + r;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0) - r;
}


float thumb(float d, vec3 p, float a, float k){
    a*=.9;
    p.yz*=rot(a);
    
    d=smin(d,length(p) -.4,.1);
    d=min(d,sdCapsule(p,vec3(0),vec3(0,-k,0),.25));
    
    
    p.y += k;
    p.yz*=rot(a);

    d=min(d,length(p) -.37);
    d=min(d, sdCapsule(p,vec3(0),vec3(0,-.9*k,0),.35));
    return d;
}

float finger(float d, vec3 p, float a, float k){
    a*=1.35;
    p.yz*=rot(a);
    
    d=smin(d,length(p) -.4, .1);
    d=min(d,sdCapsule(p,vec3(0),vec3(0,-k,0),.25));
    
    p.y += k;
    p.yz*=rot(a);

    d=min(d,length(p) -.35);
    d=min(d, sdCapsule(p,vec3(0),vec3(0,-.9*k,0),.25));
    p.y += .9*k;
    p.yz*=rot(a);

    d=min(d,length(p) -.35);
    d=min(d, sdCapsule(p,vec3(0),vec3(0,-.8*k,0),.3));
    return d;
}

float op0,op1,op2,op3,op4;
float scissor = 0.;

vec3 distorted_p;
float scene(vec3 p0) {
    float d = 999.;
    
//    d=min(d,sdEllipsoid(p0-vec3(0,2.3,-1.2),vec3(.5,.9,.9)));
//    d=min(d, sdCapsule(p0,vec3(0,2.3,-1.2),vec3(2,8,-.5),.45));
    p0 = p0.yzx;// *= rot(1.9+.2*cos(iTime));
    
    p0.zx *= rot(3.14+.2*cos(iTime));
    d = min(d, mix(sdEllipsoid(p0-vec3(1,1.3,.2), vec3(1.7,1.5,.7)),
                    sdRoundBox(p0- vec3(1,1.3,.2), vec3(1.7,1.5,.7)-.25,.25), .3));
    vec3 p1 = p0;
    p1.yx*=rot(.3*scissor);
    d = finger(d,p1, op0, .9);
    
    vec3 p2 = p0-vec3(.7,-.1,.15);
    p2.yx*=rot(-.2*scissor);
    d = finger(d,p2, op1, 1.1);
    
    d = finger(d,p0-vec3(1.4,-.1,.1), op2, 1.);
    d = finger(d,p0-vec3(2.1,0,0), op3, .8);
    
    vec3 p = p0-vec3(-.7,2.,0);
    p.xy *= rot(-.8);
    p.zx *= rot(-1.1);
    d = thumb(d,p, op4, 1.2);

    return d;
}

float softshadow(vec3 ro, vec3 rd, float mint, float k) {
    float res = 1.0;
    float t = mint;
	float h = 1.0;
    for( int i=0; i<64; i++ )
    {
        h = scene(ro + rd*t);
        res = min( res, k*h/t );
		t += clamp( h, 0.002, 2.0 );
        if( res<0.0001 ) break;
    }
    return clamp(res,0.0,1.0);
}

float calcAO(vec3 pos, vec3 nor ) {
    float ao = 0.0;
    for( int i=0; i<8; i++ )
    {
        float h = 0.02 + 0.5*float(i)/7.0;
        float d = scene(pos + h*nor);
        ao += h-d;
    }
    return clamp( 1.5 - ao*0.6, 0.0, 1.0 );
}

vec3 norm(vec3 p) {
    mat3 k = mat3(p,p,p)-mat3(0.001);
    return normalize(scene(p) - vec3(scene(k[0]),scene(k[1]),scene(k[2])));
}

mat3 setCamera( in vec3 ro, in vec3 ta, float cr) {
	vec3 cw = normalize(ta-ro);
	vec3 cp = vec3(sin(cr), cos(cr),0.0);
	vec3 cu = normalize( cross(cw,cp) );
	vec3 cv = normalize( cross(cu,cw) );
    return mat3( cu, cv, cw );
}


void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	//fragColor =vec4(fragCoord,0,0);
	//return;
#ifdef SHADERTOY
    op0 = .5+.5*cos(3.*iTime);
    op1 = .5+.5*cos(.3+3.*iTime);
    op2 = .5+.5*cos(.2+3.*iTime);
    op3 = .5+.5*cos(.1+3.*iTime);
    op4 = .5+.5*cos(.1+3.*iTime);
    scissor = .2*(.5+.5*cos(iTime));
#else
	float po = 1.5;
    op0 = 1.5*pow(1.-finger_openings[1],po);
    op1 = 1.5*pow(1.-finger_openings[2],po);
    op2 = pow(1.-finger_openings[3],po);
    op3 = pow(1.-finger_openings[4],po);
    op4 = pow(1.-finger_openings[0],po); // pouce
	op0 = clamp(op0,0.,1.);
	op1 = clamp(op1,0.,1.);
    scissor = smoothstep(.45,1.,op2-op0); 
	op4 = (op2+op3)*.5;//clamp(op4+scissor,0.,1.);
#endif	

    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y;

    float dis = 30.;
    float a = cos(.2*iTime);
    vec3 ro = dis*vec3( cos(a),.7, sin(a) );
    vec3 ta = vec3( 0. );
#ifndef SHADERTOY
	ta.y += 3.*finger_positions[0].y-1.5;
#endif	
	
        // camera-to-world transformation
    mat3 ca = setCamera( ro, ta, .1*cos(.3));

        // ray direction
    vec3 rd = ca * normalize( vec3(uv.xy,4.5) );
         
    
    float t = 0.;
    for (int i = 0; i < 250; i++) {
        float d = scene(ro+rd*t);
        t += d;
        if (t>FAR && d<1e-3) break;
    }
    
    vec3 p = ro+rd*t;
    vec3 local_coords = 2.*p; //distorted_p;
    vec3 n = norm(p);
    float ao = calcAO(p, n);
    vec3 r = reflect(rd,n);
    float ss = smoothstep(-.05,.05,scene(p+vec3(.05)/sqrt(3.)));
    float tex = .5;//WaveletNoise(local_coords*3., 0., 1.5)+.5;

   // float tex = fbm_g(local_coords);
    float diff = mix(length(sin(n*2.)*.5+.5)/sqrt(3.),ss,.7)+.1;
    float spec = length(sin(r*4.)*.5+.5)/sqrt(3.);
    float specpow = mix(3.,10.,tex);
    float frens = 1.-pow(dot(rd,n),2.)*.98;

    vec3 col,bg;
   // if (mod(iTime+.5,8.)>4.) {
        col = vec3(1.,.2,.4);
        bg = .5*vec3(.2,.3,.3);
   // } else {
   //     col = 0.*vec3(1.,.2,.4);
   //     bg = vec3(1);//2.*vec3(.3,.3,.2);
   // }
 

    col = (ao*.8+.2)*col*diff + pow(spec,specpow)*frens;

    float bgdot = length(sin(rd*3.5)*.4+.6)/sqrt(3.);

    bg = bg * bgdot + pow(bgdot, 10.)*2.;
 
    fragColor.xyz = t<FAR ? col : bg;
    fragColor = sqrt(fragColor);
    fragColor *= 1.- dot(uv,uv)*.6;
}