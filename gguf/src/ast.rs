use std::{collections::HashSet, rc::Rc};

use proc_macro2::{Literal, Span, TokenStream, TokenTree};
use proc_macro_common::{bail, format_err};
use quote::ToTokens;
use syn::{
    parenthesized,
    parse::{Parse, ParseBuffer, Parser},
    parse2, parse_macro_input, parse_quote,
    punctuated::Punctuated,
    Attribute, Data, DeriveInput, Expr, Field, GenericArgument, Ident, Meta, MetaList, Path,
    PathArguments, Token, Type, TypePath,
};

#[derive(Debug)]
pub enum AType {
    Vec(Box<AType>),
    Tensor {
        format_str: Literal,
        dims: Vec<Expr>,
    },
    Struct {
        ty: TypePath,
        init_macro: Ident,
        load_macro: Ident,
    },
    KeyValue {
        format_str: Literal,
    },
    KeyValueVec {
        format_str: Literal,
    },
}

#[derive(Debug)]
pub struct AStruct {
    pub ident: Ident,
    pub init_macro: Ident,
    pub load_macro: Ident,
    pub ty: Path,
    pub fields: Vec<AField>,
    pub root: bool,
}

#[derive(Debug)]
pub struct AField {
    pub ident: Ident,
    pub escaped_ident: Ident,
    pub format_str: Option<Literal>,
    pub ty: AType,
}

impl AField {
    fn escaped_ident(&self) -> Ident {
        Ident::new(
            format!("__{}", self.ident.to_string()).as_str(),
            Span::call_site(),
        )
    }
}

#[derive(Debug)]
enum StructAttrToken {
    HParams(Vec<(Ident, Option<Type>)>),
    LayerIdent { input: Ident, span: Span },
    Root { span: Span },
}

#[derive(Debug)]
enum FieldAttrToken {
    Name { input: Literal, span: Span },
    Dim { input: Vec<Expr>, span: Span },
    Ty(Ident),
}

#[derive(Debug)]
pub struct StructAttrs {
    pub hparams: Vec<(Ident, Option<Type>)>,
    pub root: bool,
}

#[derive(Debug, Clone)]
pub struct FieldAttrs {
    format_str: Option<Literal>,
    dim: Vec<Expr>,
    ty: Option<Ident>,
}

impl Parse for FieldAttrToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attr_name: Ident = input.parse()?;
        let _: Token![=] = input.parse()?;
        match attr_name.to_string().as_str() {
            "name" => {
                let format_str = proc_macro2::Literal::parse(input)?;
                let span = format_str.span();

                Ok(FieldAttrToken::Name {
                    input: format_str,
                    span,
                })
            }
            "dim" => {
                let content;
                parenthesized!(content in input);
                let tuple: Punctuated<syn::Expr, Token![,]> =
                    Punctuated::parse_terminated_with(&content, Expr::parse)?;
                let tuple: Vec<syn::Expr> = tuple.into_iter().collect();
                match tuple.len() {
                    0 | 5.. => bail!(String::new(), "expected one to four dimensional shape"),
                    _ => {}
                };

                Ok(FieldAttrToken::Dim {
                    input: tuple,
                    span: input.span(),
                })
            }
            "ty" => {
                let ident = Ident::parse(input)?;
                Ok(FieldAttrToken::Ty(ident))
            }
            _ => bail!(attr_name, "unknown attribute argument"),
        }
    }
}

impl TryFrom<Vec<Attribute>> for FieldAttrs {
    type Error = syn::Error;

    fn try_from(attrs: Vec<Attribute>) -> syn::Result<Self> {
        let attrs: Result<Vec<Vec<FieldAttrToken>>, syn::Error> = attrs
            .into_iter()
            .map(|attr| match attr.meta {
                Meta::Path(_) => todo!(),
                Meta::List(l) => {
                    let tokens = l.tokens.into();

                    let parser = Punctuated::<FieldAttrToken, Token![,]>::parse_separated_nonempty;
                    let attr = parser.parse2(tokens)?;

                    let attrs: Vec<FieldAttrToken> = attr.into_iter().collect();
                    Ok(attrs)
                }
                Meta::NameValue(_) => todo!(),
            })
            .collect();

        let mut res = FieldAttrs {
            format_str: None,
            dim: Vec::new(),
            ty: None,
        };

        for attrs in attrs? {
            for attr in attrs {
                match attr {
                    FieldAttrToken::Name { input, span } => {
                        if res.format_str.is_some() {
                            bail!(input, "duplicate format string");
                        }
                        res.format_str = Some(input);
                    }
                    FieldAttrToken::Ty(ty) => {
                        if res.ty.is_some() {
                            bail!(ty, "duplicate type");
                        }
                        res.ty = Some(ty);
                    }
                    FieldAttrToken::Dim {
                        input: input_,
                        span,
                    } => {
                        if !res.dim.is_empty() {
                            bail!(
                                input_.first().unwrap(),
                                "duplicate definition of dimensions"
                            );
                        }
                        res.dim = input_;
                    }
                }
            }
        }

        Ok(res)
    }
}

impl Parse for StructAttrs {
    fn parse(input: &ParseBuffer) -> syn::Result<Self> {
        let attrs: Punctuated<StructAttrToken, Token![,]> = Punctuated::parse_terminated(input)?;
        let mut res = StructAttrs {
            hparams: Vec::new(),
            root: false,
        };

        let mut hparams = HashSet::new();

        for attr in attrs {
            match attr {
                StructAttrToken::HParams(values) => {
                    for (hparam, ty) in values {
                        if !hparams.insert(hparam.to_string()) {
                            bail!(hparam, "hparam defined twice");
                        }

                        res.hparams.push((hparam, ty));
                    }
                }
                StructAttrToken::LayerIdent { input, span } => {
                    bail!(Ident::new("layer", span), "unsupported attribute");
                }
                StructAttrToken::Root { span } => {
                    if res.root {
                        bail!(Ident::new("root", span), "duplicate argument root");
                    }
                    res.root = true;
                }
            }
        }

        Ok(res)
    }
}

impl Parse for StructAttrToken {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let attr_name: Ident = input.parse()?;
        match attr_name.to_string().as_str() {
            "hparams" => {
                let _: Token![=] = input.parse()?;
                //let p: Punctuated<syn::Ident, Token![=]> = Punctuated::parse_terminated(input)?;
                let content;
                parenthesized!(content in input);

                let ident: Punctuated<Ident, Token![,]> =
                    content.parse_terminated(Ident::parse, Token![,])?;
                Ok(StructAttrToken::HParams(
                    ident.into_iter().map(|i| (i, None)).collect(),
                ))
            }
            "root" => {
                //if input.is_empty() {
                Ok(StructAttrToken::Root {
                    span: attr_name.span().into(),
                })
                //} else {
                //    bail!(attr_name, "unexpected argument to root")
                //}
            }
            _ => bail!(attr_name, "unknown attribute argument"),
        }
    }
}

#[derive(Debug)]
pub enum TensorInitMode {
    Load,
    Skip,
}
impl TryFrom<(Type, FieldAttrs)> for AType {
    type Error = syn::Error;

    fn try_from((value, attrs): (Type, FieldAttrs)) -> Result<Self, Self::Error> {
        match value {
            Type::Path(TypePath { qself, path }) => {
                let Some(ty) = path.segments.last() else {
                    bail!(TypePath { qself, path }, "invalid type path");
                };

                // todo: don't assume std::vec::Vec
                let res = match ty.ident.to_string().as_str() {
                    "Vec" => {
                        let PathArguments::AngleBracketed(args) = &ty.arguments else {
                            panic!();
                        };

                        let GenericArgument::Type(ty) = args.args.first().unwrap() else {
                            bail!(ty, "missing type argument for Vec");
                        };

                        let ty = AType::try_from((ty.clone(), attrs))?;

                        if let AType::KeyValue { format_str } = ty {
                            AType::KeyValueVec { format_str }
                        } else {
                            AType::Vec(Box::new(ty))
                        }
                    }

                    "Tensor" => {
                        let FieldAttrs {
                            format_str,
                            dim,
                            ty,
                        } = attrs;

                        // todo: no dimensions, no format_str
                        AType::Tensor {
                            format_str: format_str.unwrap(),
                            dims: dim,
                        }
                    }
                    "u8" | "i8" | "u16" | "i16" | "u32" | "i32" | "u64" | "i64" | "f32" | "f64"
                    | "String" | "str" => {
                        let FieldAttrs {
                            format_str,
                            dim,
                            ty,
                        } = attrs;
                        AType::KeyValue {
                            format_str: format_str.unwrap(),
                        }
                    }
                    n => AType::Struct {
                        ty: TypePath { qself, path },
                        init_macro: Ident::new(
                            format!("__{n}_init_macro").as_str(),
                            Span::call_site(),
                        ),
                        load_macro: Ident::new(
                            format!("__{n}_load_macro").as_str(),
                            Span::call_site(),
                        ),
                    },
                };

                return Ok(res);
            }
            ty => bail!(ty, "unexpected type"),
        }
    }
}

impl TryFrom<Field> for AField {
    type Error = syn::Error;

    fn try_from(value: Field) -> Result<Self, Self::Error> {
        let Field {
            attrs, ident, ty, ..
        } = value.clone();

        let attrs: FieldAttrs = FieldAttrs::try_from(attrs)?;

        let mut ty = AType::try_from((ty, attrs.clone()))?;

        if let AType::Tensor { format_str, dims } = &mut ty {
            *format_str = attrs.format_str.clone().unwrap();
            *dims = attrs.dim;
        }

        let Some(ident) = ident else {
            bail!(value, "struct field has no identifier");
        };

        let escaped_ident = Ident::new(format!("_{ident}").as_str(), Span::call_site());
        Ok(AField {
            ident,
            format_str: attrs.format_str,
            ty,
            escaped_ident,
        })
    }
}

impl TryFrom<(DeriveInput, StructAttrs)> for AStruct {
    type Error = syn::Error;

    fn try_from((value, sattrs): (DeriveInput, StructAttrs)) -> Result<Self, Self::Error> {
        let DeriveInput {
            ident, data, attrs, ..
        } = value.clone();
        let orig = match data {
            Data::Struct(data) => data,
            Data::Enum(_) => bail!(value, "expected struct got enum"),
            Data::Union(_) => bail!(value, "expected struct got union"),
        };

        let fields = orig
            .fields
            .iter()
            .cloned()
            .map(|f| AField::try_from(f))
            .collect::<Result<Vec<_>, syn::Error>>()?;
        let ty = Path::from(ident.clone());

        Ok(AStruct {
            init_macro: Ident::new(format!("__{ident}_init_macro").as_str(), Span::call_site()),
            load_macro: Ident::new(format!("__{ident}_load_macro").as_str(), Span::call_site()),
            ident,
            ty,
            fields,
            root: sattrs.root,
        })
    }
}
